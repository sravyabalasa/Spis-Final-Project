#Modified this code

'''
expand_mnist.py
~~~~~~~~~~~~~~~~~
- Take the 10,000 MNIST training images
- Creates an expanded set of 50,000 images
- Displaces each training image up, down, left and right, by one pixel.
- Saves the resulting file to /mnist_expanded.pkl.gz.
- Modified code to allow for user input
- Expands the testing data
- Prompts user for number used in demo
'''

from __future__ import print_function


#### Libraries

# Standard library
import _pickle as cPickle
import gzip
import os.path
import random
from PIL import Image

# Third-party libraries
import numpy as np

# Other functions
import imagePrepare as prepare

#User Prompt
var= input("Input a number between 0-9 you would like to test: ")
image= Image.open(var+".jpg")
image.show()

#Expansion of testing set
print("Expanding the MNIST testing set")

#Old set must be deleted every time
if os.path.exists("mnist_expanded.pkl.gz"):
    print("The expanded testing set already exists.  Exiting.")
else:
    f = gzip.open("mnist.pkl.gz", 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding="latin1")
    f.close()
    user_input_testing_pairs = []
    j = 0 # counter
    for x, y in zip(test_data[0], test_data[1]):
        user_input_testing_pairs.append((x, y))
        image = np.reshape(x, (-1, 28))
        j += 1
        if j % 1000 == 0: print("Expanding image number", j)
        # iterate over data telling us the details of how to
        # do the displacement
        for d, axis, index_position, index in [
                (1,  0, "first", 0),
                (-1, 0, "first", 27),
                (1,  1, "last",  0),
                (-1, 1, "last",  27)]:
            new_img = np.roll(image, d, axis)
            if index_position == "first": 
                new_img[index, :] = np.zeros(28)
            else: 
                new_img[:, index] = np.zeros(28)
            user_input_testing_pairs.append((np.reshape(new_img, 784), y))
    user_input_testing_pairs.append(prepare.imagePrepare(int(var)))
    user_input_testing_pairs = [list(d) for d in zip(*user_input_testing_pairs)]
    print("Saving expanded data. This may take a few minutes.")
    f = gzip.open("mnist_expanded.pkl.gz", "w")
    cPickle.dump((training_data, validation_data, user_input_testing_pairs), f)
    f.close()
