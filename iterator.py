import chainer
import numpy as np

"""Todo: yieldに直す
"""
class Iterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = len(dataset)
        if self.length < self.batch_size:
            raise ValueError('data_size must be larger than batch_size')
        self.iteration = 0
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        self.shuffle = shuffle
        if self.shuffle:
            self.order = np.random.permutation(self.length)
        else:
            self.order = np.arange(self.length)
        self._previous_epoch_detail = -1.

    def __next__(self):
        if not self.repeat and self.iteration * self.batch_size >= self.length:
            raise StopIteration
        start = self.iteration * self.batch_size % self.length
        end = (self.iteration + 1) * self.batch_size % self.length
        if start >= end:
            end = self.length
        # if self.padding and start < end:
        #     start = end - self.batch_size
        data = self.get_data(start, end)
        self.iteration += 1
        epoch = self.iteration * self.batch_size // self.length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
            if self.shuffle:
                self.order = np.random.permutation(self.length)
        return data
    
    def get_data(self, start, end):
        return [self.dataset[index] for index in self.order[start:end]]
    
    def reset(self):
        self.iteration = 0
        self.epoch = 0
        self.is_new_epoch = False
