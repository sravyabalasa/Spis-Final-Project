#Created own comments for understanding the implementation

# %load network.py

"""
network.py
~~~~~~~~~~
Implements the stochastic gradient descent (mini batches) algorithm
Backpropogation function allows for development of gradients through the network
Difference from network 2: Uses singular cost function - Quadratic cost function
Side: Network is a class, Network objects must be created

FUNCTIONS USED:
__init__: Initializes the network object
feedforward: Applies sigmoid function
SGD: Main function, applies stochastic gradient descent through multiple functions
update_mini_batch: Updates weights and biases for a mini-batch between layers
backprop: Returns gradient for the cost function with previous weights and biases
evaluate: Returns number of test inputs that has the correct result outputted
cost_derivative: Returns difference/vector of partial derivative of cost function
                 -Between output activations (experimental) and y (predicted)
sigmoid: Returns sigmoid for all activations of a certain neuron
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        '''
        - Creates a network object
        - Parameter:"sizes" is a list
                    Contains the number of layers + number of neurons per layer
        - Weights and biases
            - Using random module, generates random weights and biases
            - Gaussian distribution: mean 0, standard deviation/variance 1
            - Input layer has no biases, output layers have biases for computation
        - Details connections of the neurons between each layer (matrix)
        - Changes as the size goes
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        '''
        - Parameter: "a" as activation value
        - Input from one layer directly to next layer, no loop
        - Applies sigmoid function
        - Creates new activation value using previous activation, weights, biases
        - Return: "a" as new activation value

        Matrix Multiplication
        - Implemented Between each layer
        - Input*weight + b for ALL a --> Next layer
        - np.dot = w*a
        '''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        '''
        - Trains neural network using mini-batch stochastic gradient descent
        - Parameters
            - training_data: list of tuples (x,y); x = input; y = desired output
            - epochs
            - mini_batch_size
            - eta: learning rate (n)
            - test_data (OPTIONAL): network will be EVALUATED for accuracy after each epoch
        '''
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        '''
        - Updates weights and biases based on previous layer/backpropogation to one mini batch
        - Parameters:
            -mini-batch: tuples of (x,y); x = input; y = desired output
            -eta: learning rate (n)
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        '''
        - Parameters (x,y)
            - takes in tuples from each mini-batch
            - x = input
            - y = desired output
        - Returns: (nabla_b,nabla_w)
            - gradient for cost function
            - numpy arrays of weights and biases BETWEEN layers
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer #z is all the activations
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) #\ let's you continue the line
        nabla_b[-1] = delta #partial derivatives, set it to the biases
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #partial derivatives, switch (so dimensions would work), set weights
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data): #MODIFIED
        '''
        - Parameter: test_data (dataset)
        - Return: number of test inputs that has the correct result outputted
            - Index of neuron that has highest activation in the final layer 
        '''
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        print (self.feedforward(test_data[0][0])) #the numpy array itself
        print (test_results[0]) #the result that was outputted
        return sum(int(x == y) for (x, y) in test_results)
        #argmax returns the maximum argument
        #finds the output that results from the data --> one hots it!
        #selects that number --> works as proper output

    def cost_derivative(self, output_activations, y):
        '''
        - Parameters;
            - output_activations: neuron that highest activation in final layer
            - y: desired output
        - Return: Vector of partial derivatives (partial C_x) for the output activations
            - Basically a difference
        '''
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    '''
    - The sigmoid function
    - Parameters: z is all activations of a certain neuron sigmoid(wx+b) where wx+b is z
    - Logistic growth: changes slowly because of slow changes of weights and biases
    '''
    return 1.0/(1.0+np.exp(-z)) #logistic growth

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
