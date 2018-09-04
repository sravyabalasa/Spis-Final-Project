"""expand_mnist.py
~~~~~~~~~~~~~~~~~~

Take the 50,000 MNIST training images, and create an expanded set of
250,000 images, by displacing each training image up, down, left and
right, by one pixel.  Save the resulting file to
../data/mnist_expanded.pkl.gz.

Note that this program is memory intensive, and may not run on small
systems.

"""

from __future__ import print_function


'''
SHIFT AROUND TO MODIFY THE TESTING DATA, ADD NEW STUFF
'''


#### Libraries

# Standard library
import _pickle as cPickle
import gzip
import os.path
import random

# Third-party libraries
import numpy as np

# Other functions
import imagePrepare2 as prepare
print("Expanding the MNIST testing set")

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
    user_input_testing_pairs.append(prepare.imagePrepare(5))
    print (user_input_testing_pairs[-1]) #shows the user input is last in the array
    #print (len(user_input_testing_pairs))
    #random.shuffle(user_input_testing_pairs)
    user_input_testing_pairs = [list(d) for d in zip(*user_input_testing_pairs)]
    print("Saving expanded data. This may take a few minutes.")
    f = gzip.open("mnist_expanded.pkl.gz", "w")
    cPickle.dump((training_data, validation_data, user_input_testing_pairs), f)
    f.close()
