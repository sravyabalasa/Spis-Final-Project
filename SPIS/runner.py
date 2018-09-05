"""
runner.py
~~~~~~~~~
1) Data is expanded in expand_mnist.py
2) Data is evaluated in runner.py
- Network is created and run
- Network 1 or Network 2
"""

#Loads in the MNIST data
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#Sets up a network with 30 hidden neurons
'''
import network
net = network.Network([784, 10,10])
net.SGD(training_data, 30,10,3.0, test_data = test_data)
'''

#Sets up a network2 with a more specific cost function
import network2
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 5, 10, 0.1, lmbda = 5.0,evaluation_data=test_data, monitor_evaluation_accuracy=True) #EVALUATED THE TEST DATA, can do either

#Load the network & check a value
#net1=network2.load('myNetwork')
#netLoaded=net.accuracy(test_data, False)

'''
TO-DO

#NOTE: Does it need to have various ranges between 0-1 or just 0 and 1? imprep
#Finish comments
#Finish powerpoint results slide
#what is validation
'''
