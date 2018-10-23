from dlvc.models.knn import KnnClassifier

import unittest

class TestKnn(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()