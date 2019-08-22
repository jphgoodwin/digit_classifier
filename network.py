import numpy as np
import random
import mnist_loader
import pdb

# This is the network class that generates and stores the
# network architecture given input dimensions, and then
# provides operations for feeding forward and propagating
# backward through the network.
class Network(object):
    # Constructor for the class, takes sizes list for the number of neurons in each
    # layer. Generates the weight matrices and bias vectors required for the
    # activation function calculations for each layer.
    def __init__(self, sizes):
        # We first determine the number of layers from the length of the list.
        self.num_layers = len(sizes)
        # We then store the sizes list internally within the class.
        self.sizes = sizes
        # Randomly initialised bias arrays for each layer after the input layer are
        # generated and stored in a biases variable.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Randomly initialised weight matrices are generated for each layer after the
        # input layer. The dimensions of the previous layer are used to generate one
        # weight for each input to each neuron in the current layer. So for instance
        # if you had a layer with 3 neurons preceded by a layer with 2 neurons, you'd
        # have a (3, 2) weight matrix.
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]

    # Function takes an input matrix of the shape (n, m) where n must match the number
    # of inputs of the network and m is the size of the input batch. This input matrix
    # is then fed through the network and a resultant prediction returned with the
    # dimensions (x, m), where x is the size of the output vector and m is input batch
    # size.
    def feedforward(self, a):
        # Iterate through the bias vectors and weight matrices for each layer and
        # calculate the activation function for each one and pass it to the next
        # layer. The initial input matrix is used in the first hidden layer calculation.
        # e.g. result of calculation for a layer with weight matrix dimensions (3, 2) and
        # input matrix from the previous layer with dimensions (2, 6), meaning it has two
        # neurons and the batch size is 6, is: (3, 2) x (2, 6) + (3, 1) = (3, 6).
        for b, w in zip(self.biases, self.weights):
            # Calculate the activation function.
            a = sigmoid(np.dot(w, a) + b)
        return a

    # Function performs stochastic gradient descent on input training data, for the
    # given number of epochs, using the specified batch size. It then optionally
    # evaluates the model on a provided test dataset.
    def SGD(self, training_data, epochs, mini_batch_size, lr, test_data=None):
        # If a test dataset has been provided, get its length.
        if test_data: n_test = len(test_data)
        # Get the length of the training dataset.
        n = len(training_data)
        # Train the model for the specified number of epochs, and evaluate it after
        # each epoch if a test dataset has been supplied.
        for j in range(epochs):
            # Shuffle the training data at the beginning of each epoch.
            random.shuffle(training_data)
            # Divide the training set into minibatches of the specified length.
            mini_batches = [training_data[k:k+mini_batch_size]
                    for k in range(0, n, mini_batch_size)]
            # Iterate through the minibatches and update the model for each one.
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            # If test data has been provided evaluate the performance of the model.
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    # Function updates model using backpropagation on a minibatch of training examples,
    # with a specified learning rate. mini_batch is a list of (x, y) pairs where x is the
    # input vector and y is the expected output vector.
    def update_mini_batch(self, mini_batch, lr):
        # Create matrices to hold input and expected output examples from minibatch.
        X = []
        Y = []
        # Initially store the vectors in a list.
        for x, y in mini_batch:
            X.append(x)
            Y.append(y)

        # Stack vectors in a 2D matrix, where the vectors form the columns. E.g. if input vectors
        # have dimensions (784, 1) and there are 10 in the minibatch, stack them horizontally so
        # the resultant matrix has dimensions (784, 10).
        X = np.hstack(X)
        Y = np.hstack(Y)
        # pdb.set_trace()

        # Backpropagate deltas between predicted and actual values for each example back
        # through the network and calculate derivatives with respect to each weight and bias value.
        # Perform backpropagation with current example. The resultant derivatives of the loss
        # function with respect to each weight and bias value are stored in dw_total and db_total.
        # They have the same dimensions as the self.weights and self.biases - ie they contain one
        # value for each weight and bias in the network.
        db_total, dw_total = self.backprop(X, Y)

        # Adjust weights and biases according to derivatives and learning rate.
        self.weights = [w-(lr/len(mini_batch))*dwt
                for w, dwt in zip(self.weights, dw_total)]
        self.biases = [b-(lr/len(mini_batch))*dbt
                for b, dbt in zip(self.biases, db_total)]

    # Function performs backpropagation on the network given a set of inputs and expected outputs.
    # It first does a forward pass to determine what the predicted values are from the network
    # with the given inputs. The cost derivative is then taken between the predicted and expected
    # outputs and this is backpropagated throught the network, to determine derivatives of the
    # cost function with respect to each individual weight and bias.
    #
    # This function supports the calculation of multiple examples simultaneously, by passing in
    # a minibatch of input and output vectors as the columns of the x and y matrices. E.g. a
    # minibatch of 10 examples each with a (784, 1) input vector and a (10, 1) output vector would
    # give x and y the dimensions (784, 10) and (10, 10), respectively.
    def backprop(self, x, y):
        # pdb.set_trace()
        # Initialise arrays to hold backpropagated deltas for each value in bias and
        # weight matrices with zeros. The elements within these matrices will contain
        # the derivatives of the loss function with respect to each weight and bias value.
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        # Set the first activation function to be the input vector x (note this may be a matrix
        # of input vectors with an column for each example in the minibatch).
        activation = x
        # List to store the activation function for each layer. Initialise with the
        # first activation function (input vector(s) x).
        activations = [x]
        # List to store the weighted input z = wa + b values for each layer (prior to applying sigmoid).
        zs = []

        # Do a forward pass through the network calculating and storing the activation
        # functions at each layer. These will be used when backpropagating the cost.
        for b, w in zip(self.biases, self.weights):
            # Activation function has form: sigmoid(wa + b).
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # pdb.set_trace()
        # Backward pass through the network, implementing gradient descent.
        # Calculate the cost derivative between the predicted and actual results.
        # Multiply this by the derivative of the sigmoid (the gradient) to get the
        # required delta. This delta is the derivative of the cost function with
        # respect to the weighted input (z) to the final layer.
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        # Replace the bias deltas for the final layer with the newly calculated
        # deltas directly. As their are multiple examples being calculated simultaneously,
        # it is necessary to sum the columns together at this stage to reduce the matrix
        # to a single vector.
        db[-1] = delta.sum(axis=1, keepdims=True)

        # Replace the weight deltas for the final layer with the product of the delta
        # vector with the transpose of the activation vector for the penultimate layer.
        # e.g. If the final layer has 10 neurons and the penultimate layer has 30, then
        # we're calculating (10, 1) x (30, 1)T = (10, 30), which is the dimensions of the
        # weight matrix for the final layer. Note in the case of a minibatch larger than 1,
        # the above multiplication becomes (10, n) x (30, n)T = (10, 30), which as you can
        # see still produces a matrix of the correct dimensions.
        dw[-1] = np.dot(delta, activations[-2].transpose())

        # Backpropagate this delta through each layer and calculate the deltas with
        # respect to each weight and bias.
        for l in range(2, self.num_layers):
            # pdb.set_trace()
            # Extract the weighted input z for the current layer.
            z = zs[-l]
            # Calculate the derivative of the sigmoid of z.
            sp = sigmoid_prime(z)
            # Calculate the derivative of the cost function with respect to the weighted input z
            # for the current layer. 
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            # Set the bias derivatives directly from delta (summed across the columns as before).
            db[-l] = delta.sum(axis=1, keepdims=True)
            # Calculate the weight derivatives by multiplying together the delta vector and the
            # transpose of the activation vector input from previous layer (in a forward pass).
            dw[-l] = np.dot(delta, activations[-l-1].transpose())

        # Return the arrays containing the cost function derivatives with respect to the bias
        # and weight values.
        return (db, dw)

    # Function tests the model on the provided test data and returns the total number of
    # examples that were identified correctly.
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # Calculates the derivative of the cost function 0.5*(y - a)^2 which gives: (a - y).
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
        
        
# Utility function for calculating sigmoid.
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# Utility function for calculation derivative of sigmoid with respect to input vector z.
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def length(i):
    return sum(1 for e in i)

net = Network([2, 3, 1])

print('input sizes')
print(net.sizes[:])
print('biases')
print(net.biases[:])
print('weights')
print(net.weights[:])

# net2 = Network([2, 3, 3, 1])
# 
# print(net2.sizes[:])
# print(net2.biases[:])
# print(net2.weights[:])

adata = np.array([2, 3]).reshape((2, 1))
print(adata.shape)
print(adata)
print(net.feedforward(adata))

training_data, validation_data, test_data = mnist_loader.load_data()
net = Network([784, 30, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
