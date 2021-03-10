import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import os

mycmap = cm.get_cmap('jet')
mycmap.set_under('w')   

X = np.loadtxt(open("./test/X.csv", "rb"), delimiter=",")
Y = np.loadtxt(open("./test/Y.csv", "rb"), delimiter=",")
Z = np.loadtxt(open("./test/Z.csv", "rb"), delimiter=",")
print('Successfuly Loaded!')
print('Plotting...')

plt.figure(1, figsize=(13,5))
mycmap = cm.get_cmap('jet')
mycmap.set_under('w')   
plt.title('Wind Speed Flow Field')
plt.pcolormesh(X, Y, Z, shading='auto',cmap=mycmap)
plt.axis('scaled')
plt.xlabel('Local x-Direction /m')
plt.ylabel('Local y-Direction /m')
cbar = plt.colorbar()
cbar.set_label('Wind Speed /(m/s)')

plt.show()