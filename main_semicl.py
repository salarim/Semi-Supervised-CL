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
from util import save_model
from main_ce import set_loader as set_val_loader
from main_linear import validate
from networks.resnet_big import SupConResNet, LinearClassifier
from losses import SupConLoss
from continual_dataset import ClassIncremental, ContinualDataset
from memory import ReserviorMemory

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
                        help='dataloader batch_size')
    parser.add_argument('--training_batch_size', type=int, default=256,
                        help='training batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--updates_per_batch', type=int, default=1,
                        help='number of training iterations per batch')

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
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle dataloader')

    # continual dataset
    parser.add_argument('--nb_tasks', type=int, default=5, 
                        help='Number of tasks for ClassIncremental dataset')
    parser.add_argument('--unlabeled_prob', type=float, default=1.0, 
                        help='Probability of unlabeled data in dataset')
    parser.add_argument('--labeled_prob', type=float, default=0.0, 
                        help='Probability of labeled data in dataset')

    # method
    parser.add_argument('--labeled_memory_capacity', type=int, default=1000, 
                        help='The capacity of memory bank for labeled data')
    parser.add_argument('--unlabeled_memory_capacity', type=int, default=1000, 
                        help='The capacity of memory bank for unlabeled data')
    parser.add_argument('--supervised_backbone_update', action='store_true',
                        help='update backbone of networks using labeled data')

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
    parser.add_argument('--warm_epochs', type=float, default=10.0,
                        help='Warmup epochs')
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

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_tbsz_{}_temp_{}_lprob_{}_unlprob_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay, 
               opt.batch_size, opt.training_batch_size, opt.temp, opt.labeled_prob, 
               opt.unlabeled_prob, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
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
        cl_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle,
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


def divide_labeled_unlabeled(images, labels):
    labeled_indexes = labels != -1
    unlabeled_indexes = labels == -1

    labeled_images_v1 = images[0][labeled_indexes]
    labeled_images_v2 = images[1][labeled_indexes]
    unlabeled_images_v1 = images[0][unlabeled_indexes]
    unlabeled_images_v2 = images[1][unlabeled_indexes]

    labeled_labels = labels[labeled_indexes]
    unlabeled_labels = labels[unlabeled_indexes]

    return [labeled_images_v1, labeled_images_v2, labeled_labels], \
                [unlabeled_images_v1, unlabeled_images_v2, unlabeled_labels]


def extend_by_memory(memory_bank, data, extended_size):
    if data[0].shape[0] >= extended_size:
        return data
    
    samples_size = extended_size - data[0].shape[0]
    memorized_data = memory_bank.get_sample(samples_size)
    
    for i in range(len(data)):
        if memorized_data[i] is not None:
            data[i] = torch.cat([data[i], memorized_data[i]], dim=0)

    return data


def train(train_loader, memory_banks, model, classifier, criterions, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        new_labeled_data, new_unlabeled_data = divide_labeled_unlabeled(images, labels)
        
        for pass_idx in range(opt.updates_per_batch):
            # calculate bsz
            labeled_bsz = new_labeled_data[0].shape[0]
            unlabeled_bsz = new_unlabeled_data[0].shape[0]
            bsz = labeled_bsz + unlabeled_bsz

            # extend by memory
            labeled_data = extend_by_memory( 
                                            memory_banks['labeled'], 
                                            new_labeled_data,
                                            int(labeled_bsz * opt.training_batch_size / bsz)
                                            )
            unlabeled_data = extend_by_memory(
                                            memory_banks['unlabeled'], 
                                            new_unlabeled_data,
                                            int(unlabeled_bsz * opt.training_batch_size / bsz)
                                            )

            # update bsz after extending
            labeled_bsz = labeled_data[0].shape[0]
            unlabeled_bsz = unlabeled_data[0].shape[0]
            bsz = labeled_bsz + unlabeled_bsz

            labeled_images = torch.cat([labeled_data[0], labeled_data[1]], dim=0)
            labels = labeled_data[2].repeat(2)
            unlabeled_images = torch.cat([unlabeled_data[0], unlabeled_data[1]], dim=0)

            if torch.cuda.is_available():
                labeled_images = labeled_images.cuda(non_blocking=True)
                unlabeled_images = unlabeled_images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            

            # adjust learning rate
            adjust_learning_rate(opt, optimizer, epoch, idx, len(train_loader), pass_idx)
            warmup_learning_rate(opt, optimizer, epoch, idx, len(train_loader), pass_idx)

            loss = 0.0
            # compute unsupervised loss
            if unlabeled_bsz > 0:
                features = model(unlabeled_images)
                f1, f2 = torch.split(features, [unlabeled_bsz, unlabeled_bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                unsup_loss = criterions['SupConLoss'](features)
                loss += unsup_loss * unlabeled_bsz

            # compute supervised loss
            if labeled_bsz > 0:
                if opt.supervised_backbone_update:
                    features = model.encoder(labeled_images)
                    output = classifier(features)
                else:
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
            
        # add current batch to memroy banks
        memory_banks['labeled'].add(*new_labeled_data)
        memory_banks['unlabeled'].add(*new_unlabeled_data)

    return losses.avg


def get_number_of_updates(epoch, batch_id, total_batches, update_id, updates_per_batch):
    updates_per_epoch = total_batches * updates_per_batch

    nb_updates = (epoch-1) * updates_per_epoch + batch_id * updates_per_batch + update_id

    return nb_updates


def adjust_learning_rate(args, optimizer, epoch, batch_id, total_batches, update_id):
    nb_updates = get_number_of_updates(epoch, batch_id, total_batches, update_id, args.updates_per_batch)
    total_nb_updates = get_number_of_updates(args.epochs+1, 0, total_batches, 0, args.updates_per_batch)

    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * nb_updates / total_nb_updates)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, optimizer, epoch, batch_id, total_batches, update_id):
    nb_updates = get_number_of_updates(epoch, batch_id, total_batches, update_id, args.updates_per_batch)
    warmup_nb_updates = get_number_of_updates(args.warm_epochs+1, 0, total_batches, 0, args.updates_per_batch)

    if args.warm and nb_updates <= warmup_nb_updates:
        p = nb_updates / warmup_nb_updates
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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
    
    # build memory banks
    memory_banks = {'labeled': ReserviorMemory(opt.labeled_memory_capacity),
            'unlabeled': ReserviorMemory(opt.unlabeled_memory_capacity)}

    # training routine
    for epoch in range(1, opt.epochs + 1):

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, memory_banks, model, classifier, criterions, optimizer, epoch, opt)
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
