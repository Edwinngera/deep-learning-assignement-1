
#from ..dataset import Sample, Subset, ClassificationDataset
import pickle
import matplotlib.pyplot as plt
import cv2
import os

##########################dataset###########################################################################

# I added the code of dataset.py here since I got an error message when import classes from dataset

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from enum import Enum
import numpy as np
import torch

'''
A dataset sample.
  idx: index of the sample in the dataset.
  data: sample data.
  label: target label.
'''
Sample = namedtuple('Sample', ['idx', 'data', 'label'])


class Subset(Enum):
    '''
    Dataset subsets.
    '''

    TRAINING = 1
    VALIDATION = 2
    TEST = 3


class Dataset(metaclass=ABCMeta):
    '''
    Base class of all datasets.
    '''

    @abstractmethod
    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''

        pass


class ClassificationDataset(Dataset):
    '''
    Base class of image classification datasets.
    Sample data are numpy arrays of shape rows*cols (grayscale) or rows*cols*channels (color).
    Sample labels are integers from 0 to num_classes() - 1.
    '''

    @abstractmethod
    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        pass

####################################################################################################

def unpickle(file):
    #“unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        return data

class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):

        self.subval = subset.value
        self.dogscats_train_set = []
        self.dogscats_val_set = []
        self.dogscats_test_set = []
        self.dir = fdir


        if os.path.exists(fdir) == False:
            raise ValueError("Directory does not exist")
        else:

            if self.subval == 1:
                train_data = None
                train_labels = []

                for i in range(1, 5):
                    data_dic = unpickle(self.dir + "/data_batch_{}".format(i))
                    if i == 1:
                        train_data = data_dic['data']
                    else:
                        train_data = np.vstack((train_data, data_dic['data']))
                    train_labels += data_dic['labels']

                self.dogscats_train_set = self.load_dogs_cats_data(train_data, train_labels)

            if self.subval == 2:
                validation_data = None
                validation_labels = []

                data_dic = unpickle(self.dir + "/data_batch_5")
                validation_data = data_dic['data']
                validation_labels = data_dic['labels']

                self.dogscats_val_set = self.load_dogs_cats_data(validation_data, validation_labels)


            if self.subval == 3:
                test_data = None
                test_labels = []

                test_dic = unpickle(self.dir + "/test_batch")
                test_data = test_dic['data']
                test_labels += test_dic['labels']

                self.dogscats_test_set = self.load_dogs_cats_data(test_data, test_labels)
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

    def load_dogs_cats_data(self, data, labels):

        label_names_dic = unpickle(self.dir + "/batches.meta")
        label_names = label_names_dic['label_names']

        dog_label = label_names.index('dog')
        cat_label = label_names.index('cat')
        dogscats_set = []

        data = data.reshape((len(data), 3, 32, 32))  # first axis stays the same, second axis (3072 values) is split into 3, and the 1024 pixels are split into 32x32 pixels
        data = np.rollaxis(data, 1, 4)  # put 3 at the end
        labels = np.array(labels)

        for j in range(0, len(labels)):

            if (labels[j] == cat_label):
                sample = Sample(idx=j, data=data[j], label=0)
                dogscats_set.append(sample)

            if (labels[j] == dog_label):
                sample = Sample(idx=j, data=data[j], label=1)
                dogscats_set.append(sample)

        return dogscats_set

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        if self.subval==1:
            return len(self.dogscats_train_set)

        if self.subval==2:
            return len(self.dogscats_val_set)

        if self.subval==3:
            return len(self.dogscats_test_set)

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''

        if self.subval == 1:
            return self.dogscats_train_set[idx]

        if self.subval == 2:
            return self.dogscats_val_set[idx]

        if self.subval == 3:
            return self.dogscats_test_set[idx]

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        pass

data_dir = r'D:\03_Biomedical_Eng\5.Semester\DLVC\1_Assignment\Cifar_Dataset\cifar-10-batches-py'


training_set = PetsDataset(data_dir, Subset(1))
val_set = PetsDataset(data_dir, Subset(2))
test_set = PetsDataset(data_dir, Subset(3))

#Test number of samples in the individual datasets:
print(training_set.__len__())
print(val_set.__len__())
print(test_set.__len__())

#Test total number of cat and dog samples: ??

#Test image shape and type
print(test_set.__getitem__(3).data.shape)
print(test_set.__getitem__(3).data.dtype) #not sure if it is np

#Test labels of first 10 training samples
for i in range(0, 10):
    print(training_set.__getitem__(i).label)

#Make sure that color channels are in BGR order by displaying images
    #Open CV follows BGR order while Matlab follows RGB order
    cv2.imshow('img', training_set.__getitem__(4).data)
    plt.imshow(training_set.__getitem__(4).data)
    #I think my images are in RGB instead of BGR, do you know how to convert them or load them in BGR?
    #we could do it with the following code:
        #b,g,r = cv2.split(bgr_img)       # get b,g,r
        #rgb_img = cv2.merge([r,g,b])     # switch it to rgb

x = 2