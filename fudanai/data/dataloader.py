import math
import random


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset, self.bs, self.shuffle = dataset, batch_size, shuffle

    def __iter__(self):
        idxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idxs)
        for i in range(0, len(idxs), self.bs):
            batch = idxs[i:i+self.bs]
            xs, ys = zip(*(self.dataset[j] for j in batch))
            yield xs, ys
            
    def __len__(self):
        """
        返回总共会产生多少个 batch。
        """
        return math.ceil(len(self.dataset) / self.bs)
