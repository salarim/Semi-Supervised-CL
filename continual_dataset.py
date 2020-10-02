from abc import ABC, abstractmethod
import numpy as np

from torch.utils.data import Dataset


class ContinualDataset(Dataset):

    def __init__(self, dataset, indexes, task_ids, keep_targets):
        self.dataset = dataset
        self.indexes = indexes
        self.task_ids = task_ids
        self.keep_targets = keep_targets
        
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        index_in_dataset = self.indexes[index]
        img, target = self.dataset.__getitem__(index_in_dataset)
        
        task_id = self.task_ids[index]
        keep_target = self.keep_targets[index]

        if not keep_target:
            target = -1
        
        return img, target, task_id


class NonIIDMaker(ABC):

    def __init__(self, targets, labeled_prob, unlabeled_prob):
        self.targets = targets
        self.labeled_prob = labeled_prob
        self.unlabeled_prob = unlabeled_prob

    @property
    @abstractmethod
    def indexes(self):
        ...

    @property
    @abstractmethod
    def task_ids(self):
        ...

    @property
    @abstractmethod
    def keep_targets(self):
        ...


class ClassIncremental(NonIIDMaker):

    def __init__(self, targets, labeled_prob, unlabeled_prob, nb_tasks):
        super().__init__(targets, labeled_prob, unlabeled_prob)

        targets_per_task = self.divide_targets_into_tasks(np.unique(targets), nb_tasks)

        indexes_per_task = self.divide_indexes_into_tasks(targets_per_task)
        
        self.combine_indexes(indexes_per_task)

    def divide_targets_into_tasks(self, unique_targets, nb_tasks):
        increment = len(unique_targets) / nb_tasks
        if not increment.is_integer():
            raise Exception(
                f"Invalid number of tasks ({nb_tasks}) for {len(unique_targets)} classes."
            )

        increment = int(increment)
        targets_per_task = [unique_targets[i*increment:(i+1)*increment] for i in range(nb_tasks)]

        return targets_per_task

    def divide_indexes_into_tasks(self, targets_per_task):
        indexes_per_task = []

        for i, targets in enumerate(targets_per_task):
            task_indexes = None

            for target in targets:
                if task_indexes is None:
                    task_indexes = np.where(self.targets == target)[0]
                else:
                    task_indexes = np.append(task_indexes, np.where(self.targets == target)[0])

            indexes_per_task.append(task_indexes)
        
        return indexes_per_task

    def combine_indexes(self, indexes_per_task):
        indexes = None
        task_ids = None
        keep_targets = None

        for task_id in range(len(indexes_per_task)):
            task_indexes = indexes_per_task[task_id]

            task_labeled_size = int(len(task_indexes) * self.labeled_prob)
            task_unlabeled_size = int(len(task_indexes) * self.unlabeled_prob)
            task_size = task_labeled_size + task_unlabeled_size

            perm = np.random.permutation(len(task_indexes))
            task_indexes = task_indexes[perm]
            if task_labeled_size > 0:
                task_indexes = np.append(task_indexes[:task_unlabeled_size], task_indexes[-task_labeled_size:])
            else:
                task_indexes = task_indexes[:task_unlabeled_size]

            task_keep_targets = np.full(task_unlabeled_size, 0)
            task_keep_targets = np.append(task_keep_targets, np.full(task_labeled_size, 1))

            if indexes is None:
                indexes = task_indexes
                task_ids = np.full(len(task_indexes), task_id)
                keep_targets = task_keep_targets
            else:
                indexes = np.append(indexes, task_indexes)
                task_ids = np.append(task_ids, np.full(len(task_indexes), task_id))
                keep_targets = np.append(keep_targets, task_keep_targets)

        self._indexes = indexes
        self._task_ids = task_ids
        self._keep_targets = keep_targets

    @property
    def indexes(self):
        return self._indexes

    @property
    def task_ids(self):
        return self._task_ids

    @property
    def keep_targets(self):
        return self._keep_targets