# Digit classifier
## Overview
Project contains several implementations of a deep learning model for digit classification using MNIST dataset.

The initial implementaion in using numpy was built based on the network detailed in Michael Nielson's online book:
http://neuralnetworksanddeeplearning.com/ with some modification, for instance: support for simultaneous processing of 
minibatches using matrices instead of for loop iteration, and upgrades to support python3. I've also added a lot of 
comments and renamed variables to help with clarity.

There is also a re-implementation of the model using PyTorch that makes use of autograd for backpropagation and
gradient calculation. Implementing the model using PyTorch also enables easy transfer of the model onto the gpu to
decrease training time.

Included in the data/ folder is a compressed copy of the MNIST dataset, which is read in (after manual decompression)
by the mnist_loader file.

## Installation instructions
1. Clone the repository.
2. Extract the mnist data using gzip and leave the extracted pickle file in the data/ directory.
3a. Place a terminal in the directory containing the network.py file and execute the file with: `python3 network.py`
3b. Same as 3a but use command: `python3 network_pytorch.py` instead.

To alter the network dimensions and training parameters, simply edit the last two lines of network.py (see function
definitions for parameter explanations).

## Dependencies
* python3
* numpy
* pytorch (if running the pytorch implementation)
* gzip
