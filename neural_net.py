"""A multi-layer perceptron for classification of MNIST handwritten digits."""
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import *

import matplotlib.pyplot as plt

#from sklearn import datasets

#from sklearn.datasets.base import get_data_home
#print (get_data_home())

from scipy.io import loadmat

#from cmath import sqrt

from random import sample 
import math

import os
 
#if __name__ != '__main__':
#    os.chdir("/home/mkrej/dyskE/MojePrg/_Python/proby1")
    



def load_mnist_m():
    '''
    Load the digits dataset
    fetch_mldata ... dataname is on mldata.org, data_home
    load 10 classes, from 0 to 9
    '''
    #mnist = datasets.fetch_openml(name='MNIST original')
    #mnist = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

    n_train = 60000  # The size of training set
    # Split dataset into training set (60000) and testing set (10000)

    # data_train = mnist.data[:n_train]
    # target_train = mnist.target[:n_train]
    # data_test = mnist.data[n_train:]
    # target_test = mnist.target[n_train:]


    mnist_raw = loadmat("../mnist-original.mat")
    mnist = {
            "data": mnist_raw["data"].T,
            "target": mnist_raw["label"][0],
            "COL_NAMES": ["label", "data"],
            "DESCR": "mldata.org dataset: mnist-original",
            }
    
    nrow = mnist["target"].shape[0]
    
    
    newTarg = np.zeros((nrow, 10)) # - 1 

    for r in range(nrow):
        newTarg[r][int(mnist["target"][r])] = 1

    # newTarg[55000]
    # mnist["target"][55000]
        
    #newTarg = newTarg - logsumexp(newTarg, axis=1, keepdims=True)
     
    if False: 
        logsumexp(newTarg[1000])
        math.log(9 + math.e)
    
        
    mnist["target"] = newTarg
    

    if False:
        x = range(0,70000,10)
        y = mnist["target"]
        y = y[x]
        plt.plot(x, y, '-b')
        
    rSeq = sample(range(70000), 70000)
    #rSeq = range(70000)
        
        

    data_train = mnist["data"][rSeq][:n_train]
    target_train = mnist["target"][rSeq][:n_train]

    data_test = mnist["data"][rSeq][n_train:]
    target_test = mnist["target"][rSeq][n_train:]

    return (data_train.astype(np.float32), target_train.astype(np.float32),
            data_test.astype(np.float32), target_test.astype(np.float32))



def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    iinp = inputs
    for W, b in params:
        outputs = np.dot(iinp, W) + b
        iinp = np.tanh(outputs)
        
    #outputs = iinp      
    return outputs - logsumexp(outputs, axis=1, keepdims=True)
    #return outputs


def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)
    return log_prior + log_lik

def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)

 
if __name__ != '__main__':
    os.chdir("/home/mkrej/dyskE/MojePrg/_Python/proby1")
    print(os.getcwd())


if True or __name__ == '__main__':


    os.chdir(os.path.dirname(__file__))
    print(os.getcwd())
    
    from data import load_mnist
    
    #os.chdir(os.path.dirname(__file__))
    #os.chdir("/home/mkrej/dyskE/MojePrg/_Python/proby1")
    #os.getcwd()
   

    
    # Model parameters
    #layer_sizes = [784, 200, 100, 10]
    layer_sizes = [784, 200, 100, 10]
    L2_reg = 1.0

    # Training parameters
    param_scale = 0.1
    batch_size = 256
    num_epochs = 5
    step_size = 0.001

    print("Loading training data...")
    N, train_images, train_labels, test_images, test_labels = load_mnist()


    init_params = init_random_params(param_scale, layer_sizes)

    num_batches = int(np.ceil(len(train_images) / batch_size))
    
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    def objective(params, iter):
        idx = batch_indices(iter)
        return -log_posterior(params, train_images[idx], train_labels[idx], L2_reg)
    
    objective_grad = grad(objective)
    

    print("     Epoch     |    Train accuracy  |       Test accuracy  ")
    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy(params, train_images, train_labels)
            test_acc  = accuracy(params, test_images, test_labels)
            print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, test_acc))

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf)




if False:
    
    params = init_params
    params = optimized_params
    
    idx = batch_indices(200)
    inputs =  train_images[idx]
    targets = train_labels[idx]
        # Get gradient of objective using autograd.
         
    np.sum(targets * targets)

    np.sum(train_labels[batch_indices(200)] * train_labels[batch_indices(20)])
                     
    inputs.shape
    targets.shape
    
    inputs[0].shape
    
    
    nnRes = neural_net_predict(params, inputs)
    
    nnRes.shape 
    targets.shape
    
    nnRes * targets

    targets[2]
    
    a = np.array([[ 5, 1 ,3], [ 1, 1 ,1]])
    b = np.array([1, 2, 3])
    a * b

    a.shape
    b.shape
    
    params[2][0].shape
    params[0][1].shape
    
    nnRes[33]
    
    ob = objective(init_params, 0)
    og = objective_grad(init_params, 0)

    len(og)

    len(og[2])
    

    for W, b in init_params:
        print(W.shape, b.shape)
        
        
    og[0][1].shape

    neural_net_predict(init_params, inputs)
    neural_net_predict(optimized_params, inputs)
    
    btIdx  = 100
    inIdx = 141
    nnRes = neural_net_predict(optimized_params, train_images[batch_indices(btIdx)])

    # przeliczone na pstwo
    np.round_(np.e**nnRes[inIdx], 2)
    train_labels[batch_indices(btIdx)][inIdx]
    sum(np.e ** nnRes[inIdx] * train_labels[batch_indices(btIdx)][inIdx])

    
    plt.plot(np.arange(10),(np.e**nnRes[4]))
    plt.show()
    sample(range(10),10)
    
    
    seq = sample(range(10),10)
    
    tt = np.arange(10)
    tt[seq]
    
    
    
    

