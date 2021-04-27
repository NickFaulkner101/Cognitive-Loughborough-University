import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import make_interp_spline, BSpline
from sklearn import linear_model, datasets
import sys
import csv
import os


data_46 = np.load('46_degree.npz',allow_pickle=True)
data_106 = np.load('106_degree.npz',allow_pickle=True)
data_226 = np.load('226_degree.npz',allow_pickle=True)



upstream_array = np.load('upstream_array.npz',allow_pickle=True)
downstream_array = np.load('downstream_array.npz',allow_pickle=True)


def get_ct_value(speed):

        windspeed_values= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
        windspeed_values_new = np.linspace(windspeed_values.min(), windspeed_values.max(), 100)  
        ctvalues = [0.891,0.885,0.865,0.839,0.840,0.840,0.834,0.827,0.820,0.812,0.803,0.801,0.796,0.784,0.753,0.698,0.627,0.546,0.469,0.403,0.351,0.310,0.275,0.245,0.219,0.198,0.180,0.164,0.150,0.138,0.127,0.117,0.108,0.100,0.093,0.087,0.081,0.077,0.072,0.068,0.064,0.060,0.057,0.054,0.051]

        spl = make_interp_spline(windspeed_values, ctvalues, k=1)  # type: BSpline
        ct_values_smoothed = spl(windspeed_values_new)

        index_speed = min(range(len(windspeed_values_new)), key=lambda i: abs(windspeed_values_new[i]-speed))
        returned_ct = ct_values_smoothed[index_speed]

        return returned_ct













# print(data_46['array'])
ransac_lists_46 = data_46['array']
ransac_lists_226 = data_226['array']
ransac_lists_106 = data_106['array']





plt.figure(1)

for i in range(0,len(ransac_lists_46)):
        plt.plot(ransac_lists_46[i][0], ransac_lists_46[i][1], color='orangered', linewidth=1)

for i in range(0,len(ransac_lists_226)):
        plt.plot(ransac_lists_226[i][0], ransac_lists_226[i][1], color='cornflowerblue', linewidth=1)

for i in range(0,len(ransac_lists_106)):
        plt.plot(ransac_lists_106[i][0], ransac_lists_106[i][1], color='darkmagenta', linewidth=1)






import matplotlib.patches as mpatches

blue_patch = mpatches.Patch(color='cornflowerblue', label='South West Wind Dir')
magenta_patch = mpatches.Patch(color='darkmagenta', label='South East Wind Dir')
orange_patch = mpatches.Patch(color='orangered', label='North East')
black = mpatches.Patch(color='black', label='Jensen Prediction')
plt.legend(handles=[orange_patch,blue_patch,magenta_patch,black])

plt.xlabel("Upstream Wind Turbine Corrected Wind Speed (m/s)")
plt.ylabel("Downstream Wind Turbine Corrected Wind Speed (m/s)")
plt.title("RANSAC for 30 Turbine Wake Single-Wake Relationships")
plt.grid()



Ct = 0.6
rd=56
kw = 0.04
turbine_distance = 770
jensen_speed = []


# print(factor)

x = np.linspace(0,25,26)

for i in range(0,len(x)):
     
    Ct = get_ct_value(i)
    print(Ct)
    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    speed_deficit = factor*x[i]
    jensen_speed.append(speed_deficit)


plt.figure(2)

plt.scatter(upstream_array['array'],downstream_array['array'],label="turbine data",s=0.5)
plt.plot(x,jensen_speed,color='black', linewidth=4,label="Jensen Fit")
plt.xlabel("Upstream Wind Turbine Corrected Wind Speed (m/s)")
plt.ylabel("Downstream Wind Turbine Corrected Wind Speed (m/s)")
plt.title("Upstream and Downstream Turbine Wind Speeds Versus Jensen Model Prediction")
plt.legend(loc='upper left') 
plt.grid()


plt.show()






# old 

# print(i)    
#     if i <= 9:
#             Ct = 0.8
#     if i >  9 and i <= 10: 
#             Ct = 0.7
#     if i >  10 and i <= 11:
#             Ct = 0.66
#     if i >  11 and i <= 12: 
#             Ct = 0.51
#     if i >  12 and i <= 13: 
#             Ct = 0.39
#     if i >  13 and i <= 14: 
#             Ct = 0.3
#     if i >  14 and i <= 15: 
#             Ct = 0.23
#     if i >  15 and i <= 16: 
#             Ct = 0.19
#     if i >  16 and i <= 17: 
#             Ct = 0.16
#     if i >  17 and i <= 18: 
#             Ct = 0.13
#     if i >  18 and i <= 19: 
#             Ct = 0.11
#     if i >  20 and i <= 21: 
#             Ct = 0.08
#     if i >  21 and i <= 22: 
#             Ct = 0.07
#     if i >  22 and i <= 23:
#             Ct = 0.06
#     if i >  23 and i <= 24:
#             Ct = 0.06   
#     if  i >= 23:
#             Ct = 0.05  