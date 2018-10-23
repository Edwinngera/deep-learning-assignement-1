from dlvc.batches import Batch, BatchGenerator

from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset

import os

import unittest


class TestBatchGenerator(unittest.TestCase):
    def test_create_batch(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), "data"), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, 100, True)
        self.assertEqual(len(batch_set), 80)
        iter_gen = iter(batch_set)
        iter_result = next(iter_gen)
        self.assertEqual(iter_result.idx[0], 9)
        iter_result = next(iter_gen)
        self.assertEqual(iter_result.idx[0], 607)

    def test_shuffle(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), "data"), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, 100, False)
        self.assertEqual(len(batch_set), 80)
        iter_gen = iter(batch_set)
        iter_result = next(iter_gen)
        self.assertFalse(iter_result.idx[0] == 9)
        iter_result = next(iter_gen)
        self.assertFalse(iter_result.idx[0] == 607)

    def test_type_error_exception(self):
        self.assertRaises(TypeError, BatchGenerator, [1, 2, 3], 100, True)

    def test_batch_size_is_not_integer_exception(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), "data"), Subset.TEST)
        self.assertRaises(TypeError, BatchGenerator, dataset, 50.5, True)

    def test_bigger_batch_then_dataset_exception(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), "data"), Subset.TEST)
        self.assertRaises(ValueError, BatchGenerator, dataset, 5000, True)

    def test_negative_batch_size_exception(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), "data"), Subset.TEST)
        self.assertRaises(ValueError, BatchGenerator, dataset, -1, True)


if __name__ == "__main__":
    unittest.main()