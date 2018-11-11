from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
from dlvc.batches import Batch, BatchGenerator
import dlvc.ops as ops
from dlvc.models.knn import KnnClassifier
from dlvc.test import Accuracy

import os
import numpy as np


def load_dataset_into_batches(file_dir_path: str, subset: Subset, subset_size: int, shuffle: bool = False):
    op = ops.chain([
        ops.vectorize(),
        ops.type_cast(np.float32)
    ])
    dataset = PetsDataset(file_dir_path, subset)
    return BatchGenerator(dataset, subset_size, shuffle, op)


def grid_search_optimizer(train: Batch, validation: Batch, test: Batch,
                          s_scope: tuple, input_dim: int, num_classes: int):

    best_accuracy = Accuracy()
    best_k = 0
    accuracy_results = []
    for k in range(s_scope[0], s_scope[1], s_scope[2]):
        knn_classifier = KnnClassifier(k, input_dim, num_classes)
        accuracy = Accuracy()
        knn_classifier.train(train.data, train.label)
        predictions = knn_classifier.predict(validation.data)
        accuracy.update(predictions, validation.label)
        accuracy_results.append((str(accuracy.accuracy()), k))
        print("I am working on k equal " + str(k))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    best_knn_classifier = KnnClassifier(best_k, input_dim, num_classes)
    accuracy = Accuracy()
    best_knn_classifier.train(train.data, train.label)
    predictions = best_knn_classifier.predict(test.data)
    accuracy.update(predictions, test.label)
    return best_accuracy, best_k, accuracy_results


if __name__ == "__main__":
    data_dir_path = os.path.join(os.getcwd(), "data")
    # train_batch_gen = load_dataset_into_batches(data_dir_path, Subset.TRAINING, 7959)
    train_batch_gen = load_dataset_into_batches(data_dir_path, Subset.TRAINING, 2000)
    train_batch_iter = iter(train_batch_gen)
    train_batch = next(train_batch_iter)

    # validation_batch_gen = load_dataset_into_batches(data_dir_path, Subset.VALIDATION, 2041)
    validation_batch_gen = load_dataset_into_batches(data_dir_path, Subset.VALIDATION, 400)
    validation_batch_iter = iter(validation_batch_gen)
    validation_batch = next(validation_batch_iter)

    # test_batch_gen = load_dataset_into_batches(data_dir_path, Subset.TEST, 2000)
    test_batch_gen = load_dataset_into_batches(data_dir_path, Subset.TEST, 400)
    test_batch_iter = iter(test_batch_gen)
    test_batch = next(test_batch_iter)

    scope = (11, 100, 22)
    best_accuracy, best_k, search_result = grid_search_optimizer(train_batch, validation_batch, test_batch,
                                                                 scope, 3072, 2)
    for accuracy, k in search_result:
        print("Accuracy: " + str(accuracy) + " k: " + str(k) + ".")
    print("Best accuracy in equal: " + str(best_accuracy) + " for k equal: " + str(best_k) + ".")
