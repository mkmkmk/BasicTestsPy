"""

>> https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt_zi.html

"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import time

# GenBut4BP(250, 3, 20)
fsamp = 250
fparam = np.array([3.0, 20.0])
sos = signal.butter(2, fparam / fsamp * 2, btype='bandpass', output='sos')
print(sos)
# w Py 'b' (MA) pierwszej sekcji jest wymnożone przez wzmocnienie
nsos = np.copy(sos)
nsos[0, :3] /= nsos[0, 0]
print(nsos)


# Butt4_0c1_2c1_1000Hz
fsamp = 1000
fparam = np.array([0.1, 2.1])
sos = signal.butter(2, fparam / fsamp * 2, btype='bandpass', output='sos')
print(sos)

# w Py 'b' (MA) pierwszej sekcji jest wymnożone przez wzmocnienie
nsos = np.copy(sos)
nsos[0, :3] /= nsos[0, 0]
print(nsos)
print(signal.sosfilt_zi(sos))

fsamp = 1000
fparam = np.array([6, 20])
sos = signal.butter(2, fparam / fsamp * 2, "bandpass", output='sos')
zi0 = signal.sosfilt_zi(sos)
for row in sos:
    print("new[] {{ {} }},".format(", ".join("{}".format(x) for x in row )))

for row in zi0:
    print("new[] {{ {} }},".format(", ".join("{}".format(x) for x in row )))
   
print(np.array_str(sos, max_line_width=800, precision=20, suppress_small=True))  
  

  

x = (np.arange(250) < 100).astype(int)
f1 = signal.sosfilt(sos, x)


  

print("done")
