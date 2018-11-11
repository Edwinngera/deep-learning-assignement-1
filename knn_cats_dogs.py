from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
from dlvc.batches import Batch, BatchGenerator
import dlvc.ops as ops
from dlvc.models.knn import KnnClassifier
from dlvc.test import Accuracy
import matplotlib.pyplot as plt

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

    best_accuracy_valid = Accuracy()
    best_k = 0
    accuracy_results = []
    k_values = []
    for k in range(s_scope[0], s_scope[1], s_scope[2]):
        knn_classifier = KnnClassifier(k, input_dim, num_classes)
        accuracy = Accuracy()
        knn_classifier.train(train.data, train.label)
        predictions = knn_classifier.predict(validation.data)
        accuracy.update(predictions, validation.label)
        current_accuracy = accuracy.accuracy()
        accuracy_results.append(current_accuracy)
        k_values.append(k)
        print("I am working on k = " + str(k) + ' and ' + str(current_accuracy))
        if accuracy > best_accuracy_valid:
            best_accuracy_valid = accuracy
            best_k = k

    best_knn_classifier = KnnClassifier(best_k, input_dim, num_classes)
    accuracy = Accuracy()
    best_knn_classifier.train(train.data, train.label)
    predictions = best_knn_classifier.predict(test.data)
    accuracy.update(predictions, test.label)
    best_accuracy_test = accuracy.accuracy()
    return best_k, best_accuracy_valid, best_accuracy_test, accuracy_results, k_values


if __name__ == "__main__":
    data_dir_path = os.path.join(os.getcwd(), "data")
    train_batch_gen = load_dataset_into_batches(data_dir_path, Subset.TRAINING, 7959)
    train_batch_iter = iter(train_batch_gen)
    train_batch = next(train_batch_iter)

    validation_batch_gen = load_dataset_into_batches(data_dir_path, Subset.VALIDATION, 2041)
    validation_batch_iter = iter(validation_batch_gen)
    validation_batch = next(validation_batch_iter)

    test_batch_gen = load_dataset_into_batches(data_dir_path, Subset.TEST, 2000)
    test_batch_iter = iter(test_batch_gen)
    test_batch = next(test_batch_iter)


    # scope = (11, 100, 22)
    # best_accuracy, best_k, search_result = grid_search_optimizer(train_batch, validation_batch, test_batch,
    #                                                              scope, 3072, 2)
    # for accuracy, k in search_result:
    #     print("Accuracy: " + str(accuracy) + " k: " + str(k) + ".")
    # print("Best accuracy in equal: " + str(best_accuracy) + " for k equal: " + str(best_k) + ".")

    scope = (1, 101, 10)
    best_k, best_ac_valid, best_ac_test, accuracy_values, k_values = grid_search_optimizer(train_batch, validation_batch, test_batch, scope, 3072, 2)
    print("Best accuracy using validation set: " + str(best_ac_valid) + " for k equal:" + str(best_k) + ".")
    print("Best accuracy using test set: " + str(best_ac_test) + " using best k of validation set.")
    plt.figure()
    plt.title('Accuracies for different k values')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.plot(k_values, accuracy_values, 'r+')
    x = 0

# >>>>>>> refs/remotes/origin/master
