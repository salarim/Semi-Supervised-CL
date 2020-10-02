from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import save_model
from main_ce import set_loader as set_val_loader
from main_linear import validate
from networks.resnet_big import SupConResNet, LinearClassifier
from losses import SupConLoss
from continual_dataset import ClassIncremental, ContinualDataset

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--val_freq', type=int, default=50,
                        help='validation frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # continual dataset
    parser.add_argument('--nb_tasks', type=int, default=5, 
                        help='Number of tasks for ClassIncremental dataset')
    parser.add_argument('--unlabeled_prob', type=float, default=1.0, 
                        help='Probability of unlabeled data in dataset')
    parser.add_argument('--labeled_prob', type=float, default=0.0, 
                        help='Probability of labeled data in dataset')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_lprob_{}_unlprob_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.labeled_prob, 
               opt.unlabeled_prob, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_dataset(opt):
    # construct dataset
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    return train_dataset


def set_loader(opt):
    train_dataset = set_dataset(opt)
    
    targets = train_dataset.targets
    if torch.is_tensor(targets):
        targets = targets.numpy()
    elif type(targets) == list:
        targets = np.array(targets)
    
    class_incremental = ClassIncremental(
                                         targets, 
                                         opt.labeled_prob, 
                                         opt.unlabeled_prob, 
                                         opt.nb_tasks
                                        )

    cl_dataset = ContinualDataset(
                                  train_dataset, 
                                  class_incremental.indexes, 
                                  class_incremental.task_ids, 
                                  class_incremental.keep_targets
                                 )

    train_loader = torch.utils.data.DataLoader(
        cl_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    
    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    criterions = {'SupConLoss': SupConLoss(temperature=opt.temp),
                 'CrossEntropyLoss': torch.nn.CrossEntropyLoss()}

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        classifier = classifier.cuda()
        for name, criterion in criterions.items():
            criterions[name] = criterion.cuda()
        cudnn.benchmark = True

    return model, classifier, criterions


def train(train_loader, model, classifier, criterions, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = labels.shape[0]
        images = torch.cat([images[0], images[1]], dim=0)
        labels = labels.repeat(2)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        labeled_images = images[labels != -1]
        unlabeled_images = images[labels == -1]
        labels = labels[labels != -1]
        unlabeled_bsz = unlabeled_images.shape[0]
        labeled_bsz = labeled_images.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        loss = 0.0
        # compute unsupervised loss
        if unlabeled_bsz > 0:
            features = model(unlabeled_images)
            f1, f2 = torch.split(features, [int(unlabeled_bsz/2), int(unlabeled_bsz/2)], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            unsup_loss = criterions['SupConLoss'](features)
            loss += unsup_loss * unlabeled_bsz

        # compute supervised loss
        if labeled_bsz > 0:
            with torch.no_grad():
                features = model.encoder(labeled_images)
            output = classifier(features.detach())
            sup_loss = criterions['CrossEntropyLoss'](output, labels)
            loss += sup_loss * labeled_bsz  
        
        # take average of loss
        loss = loss / (unlabeled_bsz + labeled_bsz)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def set_optimizer(opt, parameters):
    optimizer = optim.SGD(parameters,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)
    _, val_loader = set_val_loader(opt)

    # build model and criterion
    model, classifier, criterions = set_model(opt)

    # build optimizer
    parameters = list(model.parameters()) + list(classifier.parameters())
    optimizer = set_optimizer(opt, parameters)

    # tensorboard
    writer = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, classifier, criterions, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # eval for one epoch
        if epoch % opt.val_freq == 0 or epoch == opt.epochs:
            loss, val_acc = validate(val_loader, model, classifier, criterions['CrossEntropyLoss'], opt)
            if val_acc > best_acc:
                best_acc = val_acc

        # tensorboard logger
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
