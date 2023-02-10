#from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd


from pandas import read_csv

import os

os.chdir("E:/MojePrg/_Python/boxplotMP")


dane1 = read_csv("90_do_85.csv", sep=';')
dane2 = read_csv("od_85.csv", sep=';')

data_to_plot = [ dane1['hr'],  dane2['hr'] ]


fig = plt.figure(1, figsize=(9, 6))


ax = fig.add_subplot(111)

bp = ax.boxplot(data_to_plot, widths = 0.6, patch_artist = True)

# plt.xlabel("no. of features")
plt.ylabel(r"$\Delta$HR [%]", fontsize=18)

print("hello")

for whisker in bp['whiskers']:
    whisker.set(color='#000000')
    
for cap in bp['caps']:
    cap.set(color='#000000')
    
for median in bp['medians']:
    median.set(color='#000000', linewidth=1.5)    
    
for box in bp['boxes']:
    box.set( facecolor = '#ffffff' )    
    
fig.tight_layout() 

    
#plt.savefig('e:/boxplot.pdf', format='pdf', dpi=1000)

print("done")

"""
jak z pythona do Word'a
- zapisac w Py jako .pdf
- otworzyc w Inkscape
- zapisac jako .emf
"""