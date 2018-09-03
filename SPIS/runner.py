#Loads in the MNIST data
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#Sets up a network with 30 hidden neurons
'''
import network
net = network.Network([784, 10,10])
net.SGD(training_data, 30,10,3.0, test_data = test_data)
'''

#Sets up a network with a more specific cost function
import network2
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
#net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)

#Saves the network after good training to file myNetwork
#net.save('myNetwork')
