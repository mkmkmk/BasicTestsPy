'''
Created on 7.10.2019

@author: mkrej
'''

import numpy as np
import matplotlib.pyplot as plt
from array import array
#from scipy.signal import butter, lfilter

import scipy.signal as signal
import time


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


def my_resample(x, q):
    b, a = signal.butter(5, .8 / q)
    zi = x[0] * signal.lfilter_zi(b, a) 
    y = signal.lfilter(b, a, x, zi = zi)[0][slice(0, len(x), q)]
    return y

start = time.clock()
#b, a = signal.butter(5, .8 / q)
# próbkowanie z filtrem
#sig2ft = signal.lfilter(b, a, sig1)[sampSeq]

sig2ft = my_resample(sig1, q)

#gn = np.std(sig2) / np.std(sig2ft)
#sig2ft = gn * sig2ft
print ("exe time: {:.3f}s".format(time.clock() - start))

start = time.clock()
sigPoly = signal.resample_poly(sig1, 1, q)
print ("exe time: {:.3f}s".format(time.clock() - start))

if True:
    plt.plot(sig2, 'r-', sig2ft, 'b-', sigPoly, 'g-')
    #plt.axis('tight')
    plt.show()


# https://github.com/scipy/scipy/blob/v1.3.1/scipy/signal/signaltools.py#L2408
down = 5
down = 4
h = signal.firwin(2 * 10 * down + 1, .8 / down, window=('kaiser', 5.0))
        
h_Q32 = np.round(h * 2**16).astype(int)
h_Q32   
       
h = signal.firwin(2 * 10 * down + 1, .8 / down)
               
        
        
        
    
        
        
        


