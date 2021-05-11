import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import os

mycmap = cm.get_cmap('jet')
mycmap.set_under('w')   

X = np.loadtxt(open("X.csv", "rb"), delimiter=",")
Y = np.loadtxt(open("Y.csv", "rb"), delimiter=",")
Z = np.loadtxt(open("Z.csv", "rb"), delimiter=",")
AEP_data = np.loadtxt(open("power_generated.csv", "rb"), delimiter=",")

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



print('Free Power: ' + str(AEP_data[0]))
print('Farm Generated Power: ' + str(AEP_data[1]))
print('Directional Power Loss: ' + str(AEP_data[2]) + '%')

plt.show()