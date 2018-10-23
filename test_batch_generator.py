from dlvc.batches import Batch, BatchGenerator

from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset

import os

import unittest


class TestBatchGenerator(unittest.TestCase):
    def test_create_batch(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), "data"), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, 100, False)
        self.assertEqual(len(batch_set), 80)
        iter_gen = iter(batch_set)
        iter_result = next(iter_gen)
        self.assertEqual(iter_result.idx[0], 9)

if __name__ == "__main__":
    unittest.main()