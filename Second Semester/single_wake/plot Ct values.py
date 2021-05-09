import matplotlib.pyplot as plt
import numpy as np

windspeed_values_old= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
# windspeed_values_new = np.linspace(windspeed_values.min(), windspeed_values.max(), 100)  
ctvalues_old = [0.891,0.885,0.865,0.839,0.840,0.840,0.834,0.827,0.820,0.812,0.803,0.801,0.796,0.784,0.753,0.698,0.627,0.546,0.469,0.403,0.351,0.310,0.275,0.245,0.219,0.198,0.180,0.164,0.150,0.138,0.127,0.117,0.108,0.100,0.093,0.087,0.081,0.077,0.072,0.068,0.064,0.060,0.057,0.054,0.051]

windspeed_values= np.array([2,2.5,3,3.25,3.375,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
ctvalues=[0,0,0,0,0,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.60,0.60,0.60,0.57,0.57,0.5,0.45,0.45,0.4,0.39,0.35,0.325,0.325,0.3,0.3,0.25,0.25,0.25,0.25,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]



plt.title('Windspeed vs Thrust Coefficient Vestas V112 3.45 MW')
plt.xlabel('Wind speed (m/s)',fontsize=16)
plt.ylabel('Thrust Coefficient ',fontsize=16)
plt.plot(windspeed_values,ctvalues,linewidth=2,color=[0.257,0.5195,0.953],label="Optimised")
plt.plot(windspeed_values_old,ctvalues_old,linewidth=2,color='red',label="OEM")
plt.legend(loc="upper right")
plt.grid()
plt.show()