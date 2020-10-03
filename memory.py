import random

import torch

class ReserviorMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.images_v1 = None
        self.images_v2 = None
        self.targets = None
        self.seen_size = 0

    def _get_memory_size(self):
        memory_size = 0
        if self.images_v1 is not None:
            memory_size = self.images_v1.shape[0]
        return memory_size

    def add(self, new_images_v1, new_images_v2, new_targets):
        memory_size = self._get_memory_size()

        mem_indexes = torch.arange(0, memory_size, dtype=torch.long)
        new_indexes = torch.arange(memory_size, memory_size + new_images_v1.shape[0], dtype=torch.long)

        for new_index in new_indexes:
            self.seen_size += 1
            if mem_indexes.shape[0] < self.capacity:
                mem_indexes = torch.cat((mem_indexes, torch.tensor([new_index])))
            else:
                s = int(random.random() * self.seen_size)
                if s < self.capacity:
                    mem_indexes[s] = new_index
            
        if self.images_v1 is None:
            self.images_v1 = new_images_v1
            self.images_v2 = new_images_v2
            self.targets = new_targets
        else:
            self.images_v1 = torch.cat((self.images_v1, new_images_v1), dim=0)
            self.images_v2 = torch.cat((self.images_v2, new_images_v2), dim=0)
            self.targets = torch.cat((self.targets, new_targets), dim=0)
        
        self.images_v1 = self.images_v1[mem_indexes]
        self.images_v2 = self.images_v2[mem_indexes]
        self.targets = self.targets[mem_indexes]

    def get_sample(self, size):
        memory_size = self._get_memory_size()

        if memory_size < size:
            return self.images_v1, self.images_v2, self.targets
        
        sample_indexes = torch.randperm(memory_size)[:size]
        
        return self.images_v1[sample_indexes], self.images_v2[sample_indexes], self.targets[sample_indexes]
    