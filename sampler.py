import numpy as np


class ClassSampler:

    def __init__(self, dataset, batch_size, replace=True):
        self.labels = np.asarray(dataset.labels)
        self.classes = np.unique(self.labels)
        self.idxs_by_class = [
            np.where(self.labels == c)[0] for c in self.classes]
        self.batch_size = batch_size
        self.replace = replace

    def __iter__(self):
        for cls_idxs in self.idxs_by_class:
            sample_idxs = np.random.choice(
                cls_idxs,
                size=self.batch_size,
                replace=self.replace)
            for idx in sample_idxs:
                yield idx
