import numpy as np


class ClassSampler:

    def __init__(self, dataset, batch_size, replace=True):
        self.labels = np.asarray(dataset.labels)
        self.classes = np.unique(self.labels)
        self.indices_by_class = [np.where(self.labels == c)[0] for c in self.classes]
        self.batch_size = batch_size
        self.replace = replace

    def __iter__(self):
        for cls_idxs in self.indices_by_class:
            sample = np.random.choice(
                cls_idxs,
                size=self.batch_size,
                replace=self.replace)
            for i in sample:
                yield i
