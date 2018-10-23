from ..model import Model
from ..batches import Batch, BatchGenerator
from ..datasets.pets import PetsDataset
from ..dataset import Subset
from .. import ops

from operator import itemgetter

import numpy as np
import os

class KnnClassifier(Model):
    """
    k nearest neighbors classifier.
    Returns softmax class scores (see lecture slides).
    """

    def __init__(self, k: int, input_dim: int, num_classes: int):
        """
        Ctor.
        k is the number of nearest neighbors to consult (>= 1).
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        """

        if not k >= 1:
            raise ValueError("The number of k-nearest neighbors must be at least 1, it is: ", k, ".")

        if not input_dim > 0:
            raise ValueError("The length of input vector must be greater then 0, it is: ", input_dim, ".")

        if not num_classes > 1:
            raise ValueError("The number of classes to classify must be must be at least 1, it is: ", num_classes, ".")

        if not np.issubdtype(type(k), np.integer):
            raise TypeError("The number of k-nearest neighbors must be at least 1, it is: ", k, ".")

        if not np.issubdtype(type(input_dim), np.integer):
            raise TypeError("The length of input vector must be greater then 0, it is: ", input_dim, ".")

        if not np.issubdtype(type(num_classes), np.integer):
            raise TypeError("The number of classes to classify must be must be at least 1, it is: ", num_classes, ".")

        self.k_n_n = k
        self.input_dim = input_dim
        self.num_classes = num_classes

    def input_shape(self) -> tuple:
        """
        Returns the expected input shape as a tuple, which is (0, input_dim).
        """
        return 0, self.input_dim

    def output_shape(self) -> tuple:
        """
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        """
        return self.num_classes,

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Train the model on batch of data.
        As training simply entails storing the data, the model is reset each time this method is called.
        Data are the input data, with shape (m, input_dim) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns 0 as there is no training loss to compute.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        """

        training_loss = 0

#############################################################################################

        knn_all_img = []

        for i in range(0, len(data)):

            distance = []
            nearest_neighbors = []
            k_nearest_neighbors = []

            for j in range(0, len(data)):
                dist = 0
                for k in range(0, len(self.input_dim)):
                    dist = dist + (data[i][k]-data[j+1][k])
                distance.append((dist, labels[j+1]))

            nearest_neighbors = sorted(distance, key=itemgetter(0))
            k_nearest_neighbors = nearest_neighbors[0:self.k_n_n](1)


            knn_all_img.append((k_nearest_neighbors.count(0),k_nearest_neighbors.count(1)))

###############################################################################################

        return training_loss



    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict softmax class scores from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, output_shape()) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        """

        def softmax(w, t=1.0):
            """Calculate the softmax of a list of numbers w.

            Parameters
            ----------
            w : list of numbers
            t : float

            Return
            ------
            a list of the same length as w of non-negative numbers

            Examples
            --------
            >>> softmax([0.1, 0.2])
            array([ 0.47502081,  0.52497919])
            >>> softmax([-0.1, 0.2])
            array([ 0.42555748,  0.57444252])
            >>> softmax([0.9, -10])
            array([  9.99981542e-01,   1.84578933e-05])
            >>> softmax([0, 10])
            array([  4.53978687e-05,   9.99954602e-01])
            """


            e = np.exp(np.array(w) / t)
            sm = e / np.sum(e)
            return sm


