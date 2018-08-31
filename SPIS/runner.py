#Loads in the MNIST data
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#Sets up a network with 30 hidden neurons
import network
net = network.Network([784, 100,10])
net.SGD(training_data, 30,10,3.0, test_data = test_data)
