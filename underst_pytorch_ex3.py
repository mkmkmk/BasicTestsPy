"""

https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

import time
import os
from matplotlib import _pylab_helpers

# $ pip install torchviz
from torchviz import make_dot


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Generation
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)

# Shuffles the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:80]
# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

plt.plot(x_val, y_val, 'r.')
plt.plot(x_train, y_train, 'b.')
if False:
    # plt.show()
    # _pylab_helpers.Gcf.get_active().canvas.start_event_loop(0)
   
    plt.pause(0)

# Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# and then we send them to the chosen device
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

# Here we can see the difference - notice that .type() is more useful
# since it also tells us WHERE the tensor is (device)
print(type(x_train), type(x_train_tensor), x_train_tensor.type())


# We can specify the device at the moment of creation - RECOMMENDED!
torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
print("BEFORE: a,b")
print(a, b)

# Sets learning rate
lr = 1e-1
# Defines number of epochs
n_epochs = 1000

# Defines a MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

# Defines a SGD optimizer to update the parameters
optimizer = optim.SGD([a, b], lr=lr)


for epoch in range(n_epochs):
    
    # Computes our model's predicted output
    yhat = a + b * x_train_tensor
    
    if False:
        # How wrong is our model? That's the error! 
        error = (y_train_tensor - yhat)
        # It is a regression, so it computes mean squared error (MSE)
        loss = (error ** 2).mean()
        
        if False:
            print("now dot...")
            os.getcwd()
            make_dot(yhat).render("torch_graph_yhat", format="png")
            make_dot(error).render("torch_graph_error", format="png")
            make_dot(loss).render("torch_graph_loss", format="png")
            break

    loss = loss_fn(y_train_tensor, yhat)

    # No more manual computation of gradients!
    # # Computes gradients for both "a" and "b" parameters
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_tensor * error).mean()

    # We just tell PyTorch to work its way BACKWARDS from the specified loss!
    loss.backward()

    if False:
        # Let's check the computed gradients...
        print(a.grad)
        print(b.grad)

    # No more telling PyTorch to let gradients go!
    if False:
        # THIRD ATTEMPT
        # We need to use NO_GRAD to keep the update out of the gradient computation
        # Why is that? It boils down to the DYNAMIC GRAPH that PyTorch uses...
        with torch.no_grad():
            a -= lr * a.grad
            b -= lr * b.grad
        # PyTorch is "clingy" to its computed gradients, we need to tell it to let it go...
        a.grad.zero_()
        b.grad.zero_()
    
    
    optimizer.step()
    
    optimizer.zero_grad()



print("AFTER: a,b")        
print(a, b)

print("done")

