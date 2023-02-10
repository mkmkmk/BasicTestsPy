# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 11:32:41 2018

@author: mkrej
"""

from numpy import array, ones
import matplotlib.pyplot as plt
from scipy.signal import lfilter, lfilter_zi, butter, lfiltic
from scipy import signal
import numpy as np
from sos_tests import fparam

b, a = butter(2, 0.001, btype='high')


a
b

zi2 = lfiltic(b, a, y = np.repeat(0, 30), x = np.repeat(1, 30) * 1001)
zi2


zi1 = lfilter_zi(b, a)*1001
zi1

butter()


np.arange(8) / 8
array

sos = butter(2, 0.001, btype='high', output='sos')

x = 1000 + np.arange(1, 5000) % 100
plt.plot(x)


# x = signal.unit_impulse(700)

y_sos = signal.sosfilt(sos, x)
plt.plot(y_sos)

t = np.arange(1,5000)
y_sos_ic = signal.sosfilt(sos, x, zi=np.mat(zi1))

plt.plot(t, y_sos,'r', t, y_sos_ic[0],'b')


#np.size(y_sos)
#plt.show()


from scipy import linalg
linalg.companion(a)

IminusA = np.eye(2) - linalg.companion(a).T

b[1:]



sos = butter(2, [0.002, 0.016], btype='pass', output='sos')

signal.sosfilt_zi(sos)*x[0]


a = [1, -2,  1]
b = [1.0000000, -1.9917666,  0.9918145]
lfilter_zi(b, a)
lfilter_zi(a, b)

lfilter_zi(sos[1,:3], sos[1,3:])







#------------------------------------
# Python
import numpy as np
from scipy import signal
fsamp = 1000 / 8
fparam = 4.0
sos = signal.butter(5, fparam / fsamp * 2, output='sos')
zi0 = signal.sosfilt_zi(sos)
for row in sos:
    print("new[] {{ {} }},".format(", ".join("{}".format(x) for x in np.append(row, [1., 1.]) )))
for row in zi0:
    print("new[] {{ {} }},".format(", ".join("{}".format(x) for x in row )))


np.append(row, [1., 1.])



    
       
    
    
    
    
    
    
