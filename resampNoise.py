'''
Created on 26 sie 2019

@author: mkrej
'''

import numpy as np
import matplotlib.pyplot as plt
from array import array
#from scipy.signal import butter, lfilter

import scipy.signal as signal


dur = 5
# duża cz. próbkowania sygnału udającego sygnał analogowy
fsamp = 100e3
fsig = 3.3333333
lng = int(dur * fsamp)


t = np.arange(0, lng) / fsamp

sig = 1/5 * np.sign(np.sin(t * 2 * np.pi * fsig))
sig = 1.1 * np.sign(np.sin(t * 2 * np.pi * fsig))

noise = np.random.randn(lng)

sig1 = noise + sig

if False:
    np.std(sig)/np.std(noise)

q = 1000
sampSeq = np.arange(0, lng, q)

# próbkowanie bez filtra
sig2 = sig1[sampSeq]

if False:
    sig1.size // sig2.size

b, a = signal.butter(5, .8 / q)

# próbkowanie z filtrem
sig2ft = signal.lfilter(b, a, sig1)[sampSeq]
gn = np.std(sig2) / np.std(sig2ft)
sig2ft = gn * sig2ft


plt.plot(sig2, 'r-', sig2ft, 'b-')
#plt.axis('tight')
plt.show()



