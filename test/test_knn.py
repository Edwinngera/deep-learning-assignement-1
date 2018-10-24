from dlvc.models.knn import KnnClassifier
from dlvc.batches import BatchGenerator
from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
import dlvc.ops as ops

import numpy as np
import os

import unittest

class TestKnn(unittest.TestCase):
    #some line
    def __init__(self, *args, **kwargs):
        super(TestKnn, self).__init__(*args, **kwargs)
        if os.path.basename(os.getcwd()) == "test":
            self._data_dir = "../data"
        else:
            self._data_dir = "data"

    def test_creation_with_proper_data(self):
        classifier = KnnClassifier(10, 3072, 2)
        self.assertEqual(classifier.input_shape(), (0, 3072))
        self.assertEqual(classifier.output_shape(), (2,))

    def test_wrong_value_of_knn(self):
        self.assertRaises(ValueError, KnnClassifier, 0, 3072, 2)

    def test_wrong_value_of_input_dim(self):
        self.assertRaises(ValueError, KnnClassifier, 10, 0, 2)

    def test_wrong_value_of_num_classes(self):
        self.assertRaises(ValueError, KnnClassifier, 10, 3072, 0)

    def test_wrong_type_of_knn(self):
        self.assertRaises(TypeError, KnnClassifier, 10.5, 3072, 2)

    def test_wrong_type_of_input_dim(self):
        self.assertRaises(TypeError, KnnClassifier, 10, 3072.5, 2)

    def test_wrong_type_of_num_classes(self):
        self.assertRaises(TypeError, KnnClassifier, 10, 3072, 2.5)

    def test_train_with_proper_data(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 7959, False, op)
        batch_iter = iter(batch_gen)
        iter_result = next(batch_iter)
        classifier = KnnClassifier(10, 3072, 2)
        classifier.train(iter_result.data, iter_result.label)

    def test_train_wrong_type_of_data(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 7959, False, op)
        batch_iter = iter(batch_gen)
        iter_result = next(batch_iter)
        classifier = KnnClassifier(10, 3072, 2)
        self.assertRaises(TypeError, classifier.train, [1, 2, 3], iter_result.label)

    def test_train_wrong_type_of_labels(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 7959, False, op)
        batch_iter = iter(batch_gen)
        iter_result = next(batch_iter)
        classifier = KnnClassifier(10, 3072, 2)
        self.assertRaises(TypeError, classifier.train, iter_result.data, [1, 2, 3])

if __name__ == "__main__":
    unittest.main()