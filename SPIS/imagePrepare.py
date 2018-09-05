'''
imagePrepare2.py
~~~~~~~~~~~~~~

- Prepares a non-MNIST image to be used by the neural network classifier
- Image is used in an expansion of a new pkl.gz file for testing data
'''

###Libraries
#Third-party libraries
import numpy as np
import cv2

def imagePrepare(number):
    # create an array where we can store our 4 pictures
    images = np.zeros((1,784))
    # and the correct values
    correct_vals = np.zeros((1,10))
    # we want to test our images which you saw at the top of this page
    i = 0
    for no in [number]:
        # read the image
        gray = cv2.imread(str(no)+".png", cv2.IMREAD_GRAYSCALE)
        # resize the images and invert it (black background)
        gray = cv2.resize(255-gray, (28, 28))
        (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # save the processed images
        cv2.imwrite(str(no)+".png", gray)
        """
        - Change images in set to range 0-1 from 0-255
        - Want a 1-D vector with 784 pixels
        NOTE: Does it need to have various ranges between 0-1 or just 0 and 1?
        """
        flatten = gray.flatten() / 255.0
        """
        - Stores flattened image
        - Generate correct_vals array (vectorized_result)
            -WHAT IS USE?
        """
        oneColumn = np.reshape(flatten, (784,1)) #Reshapes the array
        correct_val = np.zeros((10))
        correct_val[no] = 1
        correct_vals[i] = correct_val
        i += 1
        myTuple = (oneColumn,no) #numpy array to be added to testing data
    return myTuple
