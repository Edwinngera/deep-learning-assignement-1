from dlvc.pets import PetsDatasetTraining, PetsDatasetValidation, PetsDatasetTest
from dlvc.dataset import Subset

import os
import cv2 as cv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    training_set = PetsDatasetTraining(os.path.join(os.getcwd(), "data/training"), Subset.TRAINING)
    validation_set = PetsDatasetValidation(os.path.join(os.getcwd(), "data/validation"), Subset.VALIDATION)
    test_set = PetsDatasetTest(os.path.join(os.getcwd(), "data/test"), Subset.TEST)

    # Test number of samples in the individual data sets:
    print(training_set.__len__())
    print(validation_set.__len__())
    print(test_set.__len__())

    # #Test image shape and type
    print(test_set.__getitem__(3).data.shape)
    print(test_set.__getitem__(3).data.dtype) #not sure if it is np

    #Test labels of first 10 training samples
    test_samples = []
    for i in range(0, 10):
        test_samples.append(training_set.__getitem__(i).label)
    print(test_samples)

    #Make sure that color channels are in BGR order by displaying images
    #Open CV follows BGR order while Matlab follows RGB order

    bgr_training_set = []
    bgr_validation_set = []
    bgr_test_set = []

    for i in range(0, len(training_set)):
        bgr_training_set.append(cv.cvtColor(training_set.__getitem__(i).data, cv.COLOR_RGB2BGR))

    for i in range(0, len(validation_set)):
        bgr_validation_set.append(cv.cvtColor(validation_set.__getitem__(i).data, cv.COLOR_RGB2BGR))

    for i in range(0, len(test_set)):
        bgr_test_set.append(cv.cvtColor(test_set.__getitem__(i).data, cv.COLOR_RGB2BGR))

    my_little_sweet_dog = bgr_training_set.__getitem__(2)
    cv.imwrite('my_little_sweet_dog.png', my_little_sweet_dog)
    channels = cv.split(my_little_sweet_dog)
    my_little_sweet_blue_dog = channels[0]
    cv.imwrite('my_little_sweet_blue_dog.png', my_little_sweet_blue_dog)

    my_little_sweet_red_dog = channels[2]
    cv.imwrite('my_little_sweet_red_dog.png', my_little_sweet_red_dog)
    #I think my images are in RGB instead of BGR, do you know how to convert them or load them in BGR?
    #we could do it with the following code:
        #b,g,r = cv2.split(bgr_img)       # get b,g,r
        #rgb_img = cv2.merge([r,g,b])     # switch it to rgb