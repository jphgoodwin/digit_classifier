import pickle
import numpy as np
import gzip

# Load the data from the mnist pickle file and reformat it for use in our network.
def load_data():
    # Open the file.
    # f = gzip.open("data/mnist.pkl.gz", "rb")
    f = open('data/mnist.pkl', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin-1'
    print(f)
    # Load the training, validation and testing data from the pickle file. The
    # imported data should contain a tuples of the form (x, y) where x is the input
    # and y is the expected output.
    tr_d, va_d, te_d = u.load()
    # Reshape the inputs so they are vectors of the form (784, 1).
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # Convert the outputs from a scalar digit (0-9) to a 10-dimensional unit vector
    # with 1.0 in the jth position and zeroes in all other positions.
    training_results = [vectorised_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

# Function returns a 10-dimensional unit vector with a 1.0 in the jth position and
# zeroes everywhere else.
def vectorised_result(j):
    r = np.zeros((10, 1))
    r[j] = 1.0
    return r
