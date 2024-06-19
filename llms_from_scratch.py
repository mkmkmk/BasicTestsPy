"""
    Self-Attention Explained with Code
    https://medium.com/p/d7a9f0f4d94e
"""

# --------------- Ad Positional Encoding

import numpy as np;
from _signal import pause



def generate_positional_encoding(d_model, max_length, rounding):

    position = np.arange(0, max_length).reshape(max_length, 1)
    position

    even_i = np.arange(0, d_model, 2)
    denominator = 10_000**(even_i / d_model)

    even_encoded = np.round(np.sin(position / denominator), rounding)
    odd_encoded = np.round(np.cos(position / denominator), rounding)
    even_encoded
    odd_encoded

    positional_encoding = np.stack((even_encoded, odd_encoded),2).reshape(even_encoded.shape[0],-1)
    return positional_encoding


d_model = 3
max_length = 5
rounding = 2

positional_encoding = generate_positional_encoding(d_model, max_length, rounding)
positional_encoding.shape
positional_encoding

# ---------------
import matplotlib.pyplot as plt

d_model = 400
max_length = 100
rounding = 4

positional_encoding = generate_positional_encoding(d_model, max_length, rounding)


cax = plt.matshow(positional_encoding, cmap='coolwarm')
plt.title(f'Positional Encoding Matrix ({d_model=}, {max_length=})')
plt.ylabel('Position of the Embedding\nin the Sequence, pos')
plt.xlabel('Embedding Dimension, i')
plt.gcf().colorbar(cax)
plt.gca().xaxis.set_ticks_position('bottom')

plt.pause(0)

