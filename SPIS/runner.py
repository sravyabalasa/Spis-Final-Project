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
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=test_data,
    monitor_evaluation_accuracy=True) #EVALUATED THE TEST DATA, can do either

#Saves the network after good training to file myNetwork
#net.save('myNetwork')

#TO-DO

#RENAME THIS FUNCTION!!! imageprep1 or whatever
#Image constantly changes in filetype, fix it
    #Function is either imagePrepare or userInput
#Change ReadMe in file
#Clean up files
#Load,save the network
#Optimize the user input facility
#LITERALLY take in user input through the shell!
#NOTE: Does it need to have various ranges between 0-1 or just 0 and 1? imprep
#Review backpropogation --> Drew's whiteboard
#Finish comments
#datatype = float32 meaning?
#Finish powerpoint results slide


