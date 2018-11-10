
from .model import Model
from .batches import BatchGenerator

import numpy as np

from abc import ABCMeta, abstractmethod

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''
        self.accuracy = .0

        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''

        self.correctness_prediction = []

    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        if not (target < 2).all():
            raise ValueError("Targets contain unknown class.")

        if not (prediction <= 1 or prediction >= 0).all():
            raise ValueError("Prediction values must be between 0 and 1.")

        if not (prediction.shape[1] == 2):
            raise ValueError("Predicted values must have tuples of shape (2).")

        if not (prediction.shape[0] == target.shape[0]):
            raise ValueError("Prediction must have same number of values as target.")

        for i, _p in prediction:
            label = max(_p)
            if _p.index(label) == target[i]:
                self.correctness_prediction.append(1)
            else:
                self.correctness_prediction.append(0)

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''

        print ('accuracy: ' + str(self.accuracy))
        # return something like "accuracy: 0.395"


    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if not np.issubdtype(type(other), type(self.accuracy)):
            raise TypeError("Both accuracies must be of same type.")

        if other > self.accuracy:
            return True
        else:
            return False

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if not np.issubdtype(type(other), type(self.accuracy)):
            raise TypeError("Both accuracies must be of same type.")

        if other < self.accuracy:
            return True
        else:
            return False

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        self.accuracy = self.correctness_prediction.count(1)/len(self.correctness_prediction)
        # on this basis implementing the other methods is easy (one line)

        return self.accuracy
