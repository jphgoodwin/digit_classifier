import numpy as np
import random
import mnist_loader
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use gpu if available, otherwise use cpu.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Network(nn.Module):
    # Class constructor.
    def __init__(self, D_in, H_1, H_2, D_out):
        # Call parent class constructor.
        super(Network, self).__init__()

        # Create linear layers using provided dimensions.
        self.linear1 = nn.Linear(D_in, H_1)
        self.linear2 = nn.Linear(H_1, H_2)
        self.linear3 = nn.Linear(H_2, D_out)

    # Forward propagation function. The input is passed through each layer with a subsequent
    # sigmoid activation.
    def forward(self, x):
        act_1 = torch.sigmoid(self.linear1(x))
        act_2 = torch.sigmoid(self.linear2(act_1))
        act_3 = torch.sigmoid(self.linear3(act_2))
        return act_3

# Function runs training and testing cycles on input network according to specified parameters.
def run(net, training_data, epochs, mini_batch_size, lr, test_data=None):
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
                # Convert numpy arrays into torch tensors and reshape them so they are 1D
                # ie remove second dimension in (784, 1) and (10, 1).
                x_tensor = torch.from_numpy(x)
                x_tensor = x_tensor.reshape(x_tensor.size()[0]).to(device)
                y_tensor = torch.from_numpy(y)
                y_tensor = y_tensor.reshape(y_tensor.size()[0]).to(device)
                # Add tensors to the list.
                X.append(x_tensor)
                Y.append(y_tensor)
    
            # Stack vectors in a 2D matrix, where the vectors form the rows, eg (10, 784) and (10, 10).
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
                    param -= lr * param.grad
        

        # pdb.set_trace()
        # Evaluate network on test_data.
        if test_data:
            with torch.no_grad():
                test_results = [(torch.argmax(net(torch.from_numpy(x).transpose(0,1).to(device))).item(), y) for (x, y) in test_data]
                num_correct = sum(int(x == y) for (x, y) in test_results)
                print("Epoch {}".format(j))
                print("Test result sample: \n{0}".format(test_results[0:10]))
                print("Total correct: {0} / {1}".format(num_correct, n_test))

# Load data.
training_data, validation_data, test_data = mnist_loader.load_data()

# Create network instance.
net = Network(784, 100, 100, 10)

# Move network onto GPU.
net = net.to(device)

# Train and test network.
run(net, training_data, 50, 10, 0.003, test_data=test_data)


