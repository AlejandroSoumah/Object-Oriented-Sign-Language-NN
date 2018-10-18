import h5py
import numpy as np
import tensorflow as tf
import math

def load_dataset(train_dataset_Path,test_dataset_path):
    train_dataset = h5py.File(train_dataset_Path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(test_dataset_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# GRADED FUNCTION: initialize_parameters

def initialize_parameters_parametized(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network --> Array/List

    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable("W"+str(l),[layer_dims[l], layer_dims[l-1]],initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters['b' + str(l)] = tf.get_variable("b"+str(l),[layer_dims[l], 1],initializer=tf.zeros_initializer)

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters

def predict(X,parameters,layers_dims):

    L=len(layers_dims)

    params={}
    for l in range(1,L):
        params["W"+str(l)]=tf.convert_to_tensor(parameters["W"+str(l)])
        params["b"+str(l)]=tf.convert_to_tensor(parameters["b"+str(l)])
    x = tf.placeholder("float", [12288, 1])

    Z_L_Iteration = forward_propagation_for_predict(x,params,layers_dims)
    p = tf.argmax(Z_L_Iteration)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})

    return prediction

def forward_propagation_for_predict(X, parameters,layers_dims):
        Z_Parameters={

        }
        L=len(layers_dims)
        A=[]
        Z_L_Iteration="0"
        A.append(X)
        for l in range(1,L):
            #Z=tf.add(tf.matmul(parameters["W"+str(l)],A[l-1]),parameters["b"+str(l)])
            Z_Parameters["Z"+str(l)]=tf.add(tf.matmul(parameters["W"+str(l)],A[l-1]),parameters["b"+str(l)])
            #Z.append(Z_Parameters)
            #Z.append(tf.add(tf.matmul(parameters["W"+str(l)],A[l-1]),parameters["b"+str(l)]))
            #Z_Prameters
            A.append(tf.nn.relu(Z_Parameters["Z"+str(l)]))

            if l == L-1:
                Z_L_Iteration=Z_Parameters["Z"+str(l)]
        #return forward_propagation["Z"+str(3)]

        return Z_L_Iteration
def ParameterExportation(NumberOfWeights , NumberofBiases , PathWeights , PathBiases , WeightName , BiasesName ):
        for i in range( 1 , NumberOfWeights ):
            np.savetxt( PathWeights + WeightName + str(i) + '.dat' , parameters["W"+str(i)] , delimiter=',')
        for i in range( 1 , NumberofBiases ):
            np.savetxt( PathBiases + BiasesName + str(i) +'.dat', parameters["b"+str(i)] , delimiter=',')
        print(" Weights and Biases have been exported correctly !!")

def ParameterImportation(NumberofParameters,Path,NameOfWeights,NameOfBiases):
    for i in range(1,NumberofParameters):
        Neurones={
            "W" + str(i) :np.loadtxt(Path+NameOfWeights+str(i)+'.dat', delimiter=',', dtype='float32'),
            "b" + str(i) :np.array([np.loadtxt(Path+NameOfBiases+str(i)+'.dat', delimiter=',', dtype='float32')]).T
        }
        return Neurones
        print(" Weights and Biases have been imported correctly !!")
#ParameterImportation(4,'C:\\Users\\aleja\\Desktop\\DeepLearning\\Parameters\\SignDetection\\','parameters_W','parameters_b')
