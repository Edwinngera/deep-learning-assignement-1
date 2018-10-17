from ..dataset import Sample, Subset, ClassificationDataset

import pickle
import numpy as np
import os


def unpickle(file):
    # “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        return data

class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):

        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape 32*32*3, in BGR channel order.
        '''



        self.dir = fdir
        self._data_set = []
        self._class_number = 2
        self._train_file_number = 4
        self._validation_file_number = 1
        self._test_batch_file_number = 1
        self._subset_type = subset

        self._training_files = [self.dir + "/data_batch_{}".format(i) for i in range(1, self._train_file_number+1)]
        self._validation_files = self.dir + "/data_batch_5"
        self._test_files = self.dir + "/test_batch"

        if not os.path.exists(fdir):
            raise ValueError("Directory: ",fdir," does not exist")
        else:
            if self._subset_type == Subset.TRAINING:
                train_labels = []
                for i in range(1, self._train_file_number+1):
                    file_path = self.dir + "/data_batch_{}".format(i)
                    if not os.path.exists(file_path):
                        raise ValueError("File: ", file_path, " does not exist")
                    data_dic = unpickle(file_path)
                    if i == 1:
                        self._data_set = data_dic['data']
                    else:
                        self._data_set = np.vstack((self._data_set, data_dic['data']))
                    train_labels += data_dic['labels']
                self._data_set = self.load_dogs_cats_data(self._data_set, train_labels)

            elif self._subset_type == Subset.VALIDATION:
                validation_data = None
                validation_labels = []
                data_dic = unpickle(self.dir + "/data_batch_5")
                self._data_set = data_dic['data']
                validation_labels = data_dic['labels']
                self._data_set = self.load_dogs_cats_data(self._data_set, validation_labels)

            elif self._subset_type == Subset.TEST:
                test_data = None
                test_labels = []
                test_dic = unpickle(self.dir + "/test_batch")
                self._data_set = test_dic['data']
                test_labels += test_dic['labels']
                self._data_set = self.load_dogs_cats_data(self._data_set, test_labels)

    def load_dogs_cats_data(self, data, labels):

        label_names_dic = unpickle(self.dir + "/batches.meta")
        label_names = label_names_dic['label_names']

        dog_label = label_names.index('dog')
        cat_label = label_names.index('cat')
        self._class_number = 2

        dogs_cats_set = []

        data = data.reshape((len(data), 3, 32, 32))     # first axis stays the same, second axis (3072 values) is split into 3, and the 1024 pixels are split into 32x32 pixels
        for image in data:
            image[[0, 2]] = image[[2, 0]]               # RGB --> BGR
        data = np.rollaxis(data, 1, 4)                  # put 3 at the end
        labels = np.array(labels)

        for j in range(0, len(labels)):

            if (labels[j] == cat_label):
                sample = Sample(idx=j, data=data[j], label=0)
                dogs_cats_set.append(sample)

            if (labels[j] == dog_label):
                sample = Sample(idx=j, data=data[j], label=1)
                dogs_cats_set.append(sample)

        return dogs_cats_set

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        return len(self._data_set)

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''

        return self._data_set[idx]

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        return self._class_number