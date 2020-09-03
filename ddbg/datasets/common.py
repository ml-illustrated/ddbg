import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import BatchSampler

'''
# via https://zhuanlan.zhihu.com/p/80695364
from prefetch_generator import BackgroundGenerator

class DataLoaderPrefetch(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), 4)
'''

def DatasetWithIndices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """	

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    } )

class DatasetSubset(Subset):
    def __init__(self, dataset, indices):
        super(DatasetSubset, self).__init__(dataset, indices)

    def __getitem__(self, idx):
        # return original index, not access index
        orig_idx = self.indices[idx]
        item = self.dataset[ orig_idx ]
        return item[0], item[1], orig_idx

    @property
    def train_labels( self ):
        if type( self.dataset.targets ) == list:
            targets = torch.tensor( self.dataset.targets )
        else:
            targets = self.dataset.targets
        return targets[ self.indices ]


class DatasetWithIDsBase( Dataset ):

    def __init__( self, Parent_dataset_class ):
        self.Parent_dataset_class = DatasetWithIndices( Parent_dataset_class )

    
    def get_datasets(self, data_dir, train_transform, test_transform, train_subset_indicies=None, download=True):
        
        train_set = self.Parent_dataset_class(
            data_dir,
            download=download,
            train=True,
            transform=train_transform)

        test_set = self.Parent_dataset_class(
            data_dir,
            download=download,
            train=False,
            transform=test_transform)

        if type( train_subset_indicies ) != type( None ):
            train_set_subset = DatasetSubset( train_set, train_subset_indicies )
            train_set = train_set_subset

        return train_set, test_set

    def get_dataloaders(self, data_dir, batch_size=128, num_workers=4, shuffle=True, train_subset_indicies=None, download=True):
        train_transform = self.get_train_transform()
        test_transform = self.get_test_transform()
        
        train_set, test_set = self.get_datasets(
            data_dir,
            train_transform,
            test_transform,
            train_subset_indicies=train_subset_indicies,
            download=download,
        )

        # train_loader = DataLoaderPrefetch(
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

        test_loader = DataLoader(
            test_set,
            batch_size=int(batch_size/2),
            shuffle=False,
            num_workers=int(num_workers/2))

        return train_loader, test_loader
    

# from https://github.com/adambielski/siamese-triplet
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels.numpy() if type( labels ) == torch.Tensor else np.array( labels )
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
    
