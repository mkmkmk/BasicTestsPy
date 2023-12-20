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


model = nn.Linear(1, 1).to(device)

print("BEFORE: a,b")
print(model.state_dict())


# Sets learning rate
lr = 1e-1
# Defines number of epochs
n_epochs = 1000

# Defines a MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

# Defines a SGD optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)



for epoch in range(n_epochs):
    
    # sets the model to training mode
    model.train()
    
    # Computes our model's predicted output
    yhat = model(x_train_tensor)
    
    # Computes loss
    loss = loss_fn(y_train_tensor, yhat)

    # Computes gradients
    loss.backward()
    
    # Updates parameters and zeroes gradients
    optimizer.step()
    optimizer.zero_grad()



print("AFTER: a,b")        
print(model.state_dict())

print("done")

