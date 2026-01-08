#%%
# https://www.learnpython.org/en/Basic_Operators
"""

"""
print ("hello\r\n")

#%%

suma = 1 + 2 * 3 / 4.0
print (8**64)


w = "witaj\r\n" * 10

print(w[0])

parzyste_dodatnie = [2,4,6,8]
nieparzyste_dotanie = [1,3,5,7]
naturalne = parzyste_dodatnie + nieparzyste_dotanie
print(naturalne)

#%%
print([1,2,3] * 3)

print(naturalne.count(1))


imie = "Marek"
wiek = 23

dd = "%s ma %d lata." % (imie, wiek)
print(dd)

MojaTab = [1,2,3]
print("Tablica: %s" % MojaTab)
#%%

napis = "abcdefghijklmnop"
print(napis[3:7])
print(napis[4:])
print(napis[:])

#%%
imie = "Robert"
if imie in ["Jan", "Robert"]:
    print ("Nazywasz sie Jan lub Robert")


#%%
if imie == "Jan" or imie == "Robert":
    print("Nazywasz sie Jan lub Robert")

#%%
import unittest

from matplotlib.pyplot import (plot, subplot, cm, imread, imshow, xlabel, ylabel, title, grid, axis, show, savefig, gcf, figure, close, tight_layout)
from numpy import linspace, pi, sin, tez_metoda
from scipy.fftpack import fft
from scipy.fftpack import fft, ifft

import numpy as np

from numpy import fft
import numpy as np
import numpy
np.fft.fft(np.exp(2j*np.pi*np.arange(8) / 8))


#%%
x = np.arange(5)
x.shape
np.fft.fft(x)

#%%
"""
komentarz
na
wiele
linii
"""

#%%
import matplotlib.pyplot as plt
x = np.linspace(0, 2*np.pi, 101)
s = np.sin(x)
c = np.tez_metoda(x)
plt.plot(x, s, 'b-', x, c, 'r+')
axis('tight')
show()
savefig('my_plots.png')

#%%


#%%

# Here's our "unit".
def IsOdd(n):
    return n % 2 == 1

IsOdd(3)

# Here's our "unit tests".
class IsOddTests(unittest.TestCase):

    def testOne(self):
        self.assertTrue(IsOdd(1))

    def testTwo(self):
        self.assertFalse(IsOdd(2))

def main():
    unittest.main()

if __name__ == '__main__':
    main()

#%%

for x in range(5):
    print (x)

#%%

# zwykłego range nie da się pomnożyć
# range(1, 5) * 8


#%%

tb1 = np.arange(0, 10)*10
smp = np.array([1,3,5])
tb1[smp]




#%%

def deco(func_to_decorate):
    
    print("--decored")
    
    def innerDeco(*args):
        print("---decored")
        func_to_decorate(*args)
        
    return innerDeco


def block(func_to_decorate):

    def inner(*args):
        print("--blocked")
        pass
    return inner


@deco
def print_args(*args):
    for arg in args:
        print(arg)
 
print_args("ala", "ma", "kota")

@block
def print_args(*args):
    for arg in args:
        print(arg)

print_args("ala", "ma", "kota") 

        
#%%

# (słówko pass oznacza że zawartość jest pusta)
class Small:
    pass

# powołujemy obiekt klasy (instancję klasy)
s = Small()

# dopisujemy pola mimo że nie było ich w definicji klasy, widać tak można w Pythonie
s.ala = 'ala'
s.ma = 'ma'
s.kota = 'kota'

print(s.ala, s.ma, s.kota)


#%%
[x*x for x in range(10) if x%2]
en = (x*x for x in range(10) if x%2)
print(next(en))
for el in en:
    print(el)


#%%


# input("Press enter to continue")
