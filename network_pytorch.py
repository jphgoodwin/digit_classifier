import numpy as np
import random
import mnist_loader
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self, sizes):
        super(Network, self).__init__()

        self.sizes = sizes
        self.layers = []

        # for i in range(1, len(sizes)):
            # self.layers.append(nn.Linear(sizes[i-1], sizes[i]))
        self.linear1 = nn.Linear(784, 30)
        self.linear2 = nn.Linear(30, 30)
        self.linear3 = nn.Linear(30, 10)

    def forward(self, x):
       #  for l in self.layers:
       #      x = torch.sigmoid(l(x))
       #  return x
       return torch.sigmoid(self.linear3(torch.sigmoid(self.linear2(torch.sigmoid(self.linear1(x))))))

    def run(self, training_data, epochs, mini_batch_size, lr, test_data=None):
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
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # Iterate through the minibatches and update the model for each one.
            for mini_batch in mini_batches:
                # Create matrices to hold input and expected output examples from minibatch.
                X = []
                Y = []
                # Initially store the vectors in a list.
                for x, y in mini_batch:
                    x_tensor = torch.from_numpy(x)
                    x_tensor = x_tensor.reshape(x_tensor.size()[0])
                    y_tensor = torch.from_numpy(y)
                    y_tensor = y_tensor.reshape(y_tensor.size()[0])
                    X.append(x_tensor)
                    Y.append(y_tensor)

                # Stack vectors in a 2D matrix, where the vectors form the rows.
                X = torch.stack(X, dim=0)
                Y = torch.stack(Y, dim=0).float()

                pdb.set_trace()
                # Do a forward pass of X through the network to generate a Y_pred value.
                Y_pred = self(X)

                # Calculate the loss using the Mean Squared Error (MSE) between Y_pred and Y.
                loss = (Y_pred - Y).pow(2).sum()

                # Zero gradients before running the backward pass.
                self.zero_grad()

                # Do a backward pass using autograd to calculate gradient of loss with respect to
                # weight and bias Tensors within network.
                loss.backward()

                # Update weights and biases listed as model parameters.
                with torch.no_grad():
                    for param in self.parameters():
                        param -= lr * param.grad

training_data, validation_data, test_data = mnist_loader.load_data()
net = Network([784, 30, 30, 10])
# net.run(training_data, 30, 10, 3.0, test_data=test_data)

epochs = 30
mini_batch_size = 10
lr = 0.03
n = len(training_data)

# Train the model for the specified number of epochs, and evaluate it after
# each epoch if a test dataset has been supplied.
for j in range(epochs):
    # Shuffle the training data at the beginning of each epoch.
    random.shuffle(training_data)
    # Divide the training set into minibatches of the specified length.
    mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
    # Iterate through the minibatches and update the model for each one.
    for mini_batch in mini_batches:
        # Create matrices to hold input and expected output examples from minibatch.
        X = []
        Y = []
        # Initially store the vectors in a list.
        for x, y in mini_batch:
            x_tensor = torch.from_numpy(x)
            x_tensor = x_tensor.reshape(x_tensor.size()[0])
            y_tensor = torch.from_numpy(y)
            y_tensor = y_tensor.reshape(y_tensor.size()[0])
            X.append(x_tensor)
            Y.append(y_tensor)

        # Stack vectors in a 2D matrix, where the vectors form the rows.
        X = torch.stack(X, dim=0)
        Y = torch.stack(Y, dim=0).float()

        # Do a forward pass of X through the network to generate a Y_pred value.
        Y_pred = net(X)

        # Calculate the loss using the Mean Squared Error (MSE) between Y_pred and Y.
        loss = (Y_pred - Y).pow(2).sum()

        # Zero gradients before running the backward pass.
        net.zero_grad()

        # Do a backward pass using autograd to calculate gradient of loss with respect to
        # weight and bias Tensors within network.
        loss.backward()

        # Update weights and biases listed as model parameters.
        with torch.no_grad():
            for param in net.parameters():
                param -= (lr/len(mini_batch)) * param.grad
        

    # pdb.set_trace()
    # Evaluate network on test_data.
    with torch.no_grad():
        test_results = [(torch.argmax(net(torch.from_numpy(x).transpose(0,1))), y) for (x, y) in test_data]
        num_correct = sum(int(x.item() == y) for (x, y) in test_results)
        print("Test result sample: \n{0}".format(test_results[0:10]))
        print("Total correct: {0}".format(num_correct))




