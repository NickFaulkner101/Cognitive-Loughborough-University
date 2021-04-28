import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import make_interp_spline, BSpline
from sklearn import linear_model, datasets
import sys
import csv
import os
from sklearn.metrics import mean_squared_error

data_46 = np.load('46_degree_2wakes.npz',allow_pickle=True)
data_106 = np.load('106_degree_2wakes.npz',allow_pickle=True)
data_226 = np.load('226_degree_2wakes.npz',allow_pickle=True)



upstream_array = np.load('upstream_array.npz',allow_pickle=True)
downstream_array = np.load('downstream_array.npz',allow_pickle=True)




def get_ct_value(speed):

        #modified values
        windspeed_values= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
        windspeed_values_new = np.linspace(windspeed_values.min(), windspeed_values.max(), 100)  
        ctvalues = [0.691,0.685,0.665,0.639,0.640,0.640,0.634,0.627,0.620,0.612,0.603,0.601,0.696,0.684,0.53,0.698,0.627,0.546,0.469,0.403,0.391,0.350,0.3,0.355,0.219,0.298,0.280,0.264,0.250,0.238,0.227,0.217,0.208,0.200,0.193,0.187,0.181,0.177,0.172,0.168,0.164,0.160,0.157,0.154,0.151]

        #original actual turbine values
        # windspeed_values= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
        # windspeed_values_new = np.linspace(windspeed_values.min(), windspeed_values.max(), 100)  
        # ctvalues = [0.891,0.885,0.865,0.839,0.840,0.840,0.834,0.827,0.820,0.812,0.803,0.801,0.796,0.784,0.753,0.698,0.627,0.546,0.469,0.403,0.351,0.310,0.275,0.245,0.219,0.198,0.180,0.164,0.150,0.138,0.127,0.117,0.108,0.100,0.093,0.087,0.081,0.077,0.072,0.068,0.064,0.060,0.057,0.054,0.051]


        spl = make_interp_spline(windspeed_values, ctvalues, k=1)  # type: BSpline
        ct_values_smoothed = spl(windspeed_values_new)

        index_speed = min(range(len(windspeed_values_new)), key=lambda i: abs(windspeed_values_new[i]-speed))
        returned_ct = ct_values_smoothed[index_speed]

        return returned_ct





def return_power_value(speed,powercurve_windspeed_new,power_smooth):
    index_power = min(range(len(powercurve_windspeed_new)), key=lambda i: abs(powercurve_windspeed_new[i]-speed))
    power_value = power_smooth[index_power]
    return power_value
    # power.append(power_value)







# print(data_46['array'])
lists_46 = data_46['array']
lists_226 = data_226['array']
lists_106 = data_106['array']





plt.figure(1)

for i in range(0,len(lists_46)):
        plt.scatter(lists_46[i][2], lists_46[i][3], color='orangered', s=1,label='46 Degrees Data')

for i in range(0,len(lists_226)):
        plt.scatter(lists_226[i][2], lists_226[i][3], color='cornflowerblue', s=1,label='226 Degrees Data')

for i in range(0,len(lists_106)):
        plt.scatter(lists_106[i][2], lists_106[i][3], color='darkmagenta', s=1,label='106 Degrees Data')

plt.xlabel("Upstream Wind Turbine Corrected Wind Speed (m/s)")
plt.ylabel("Downstream Wind Turbine Corrected Wind Speed (m/s)")
plt.title("30 Turbine wind speed data, for rear turbine in two wakes")
plt.grid()

# plt.show()


Ct = 0.6



# print(factor)

x = np.linspace(0,25,26) # these are the speeds
jensen_speed_upstream = []
single_wake_data = []

for i in range(0,len(x)):
    rd=56
    kw = 0.04
    turbine_distance = 1440
    Ct = get_ct_value(i)
    print(Ct)
    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    jensen_speed_upstream.append(factor)
    single_wake_data.append(factor*x)

jensen_speed_mid_turbine = []

for i in range(0,len(x)):
    rd=56
    kw = 0.04
    turbine_distance = 770 #distance from mid turbine to downwind turbine
    Ct = get_ct_value(i)
    print(Ct)
    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    
    jensen_speed_mid_turbine.append(factor)

overall_jensen_speed = []
for i in range(0,len(x)):
    upstream_factor = jensen_speed_upstream[i]
    midstream_factor = jensen_speed_mid_turbine[i]
    one_minus_overall_factor = np.sqrt(np.square(1 - upstream_factor)+np.square(1 - midstream_factor)) #Root Sum of Squares method
    overall_factor = 1 -one_minus_overall_factor
    overall_jensen_speed.append(overall_factor*i)


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
plt.plot(x,overall_jensen_speed,color='black', linewidth=4,label="Jensen Fit")
# plt.plot(x,jensen_speed,color='red', linewidth=4,label="Jensen Fit")
# plt.plot(x,single_wake_data,color='red', linewidth=4,label="Jensen Fit")
plt.xlabel("Upstream Wind Turbine Corrected Wind Speed (m/s)")
plt.ylabel("Downstream Wind Turbine Corrected Wind Speed (m/s)")
plt.title("Upstream Vs Downstream wind speed for turbine in 2 wakes")
plt.legend(loc='upper left') 
plt.grid()


x_new = np.linspace(x.min(), x.max(), 100)

spl = make_interp_spline(x, overall_jensen_speed, k=1)  # type: BSpline
#powercurve load in
smooth_jensen_speed = spl(x_new)

plt.figure(200)
plt.plot(x_new,smooth_jensen_speed,linewidth=10,label="Jensen Fit")
plt.plot(x,overall_jensen_speed,color='black', linewidth=5,label="Jensen Fit")


# find closest jensen wind speed, then difference it for residual

# an upwind speed has a downwind real value, and a predicted value
#find difference between downwind and predicted value 
upstream_array = upstream_array['array']
standard_deviation = np.std(np.array(downstream_array['array']))
residuals = []
for i in range(0,len(upstream_array)):
    #corresponding downstream value
    downstream_value = downstream_array['array'][i]
    #corresponding predicted value
    upstream_value = upstream_array[i]
    index_speed = min(range(len(x_new)), key=lambda i: abs(x_new[i]-upstream_value))
    corresponding_predicted_jensen = smooth_jensen_speed[index_speed]
    normalised_difference = float(downstream_value - corresponding_predicted_jensen)/standard_deviation
    residuals.append(float(normalised_difference))
    print(str(i)+' Calculating Residuals')

import scipy.stats as stats
print(np.array(residuals))
plt.figure(1000)
stats.probplot(residuals, dist="norm", plot=plt)

plt.title("Q-Q Plot of Residuals from Jensen Prediction, Wind Speed")
plt.grid()



upstream_power_226 = np.load('226_power_up_2wakes.npz',allow_pickle=True)['array']
upstream_power_46 = np.load('46_power_up_2wakes.npz',allow_pickle=True)['array']
upstream_power_106 = np.load('106_power_up_2wakes.npz',allow_pickle=True)['array']

downstream_power_106 = np.load('106_power_down_2wakes.npz',allow_pickle=True)['array']
downstream_power_46 = np.load('46_power_down_2wakes.npz',allow_pickle=True)['array']
downstream_power_226 = np.load('226_power_down_2wakes.npz',allow_pickle=True)['array']


jensen_x_axis = np.load('jensen_power_x_2wakes.npz',allow_pickle=True)['array']
jensen_y_axis = np.load('jensen_power_y_2wakes.npz',allow_pickle=True)['array']


#for each point of upwind wind speed, get the corresponding predicted down wind windspeed (graph 1 of predicted wind speed vs actual)
# for each predicted wind speed, map this to a power. and plot this power as downwind power (graph 2, upwind power vs downind (actual and predicted via Jensen))



upstream_power = []
downstream_power = []

for i in range(0,len(upstream_power_106)):

    print('pump action yoghurt rifle')
    upstream_power.extend(upstream_power_106[i])
    downstream_power.extend(downstream_power_106[i])

# for i in range(0,len(upstream_power_46)):

#     print('2nd')
#     upstream_power.extend(upstream_power_46[i])
#     downstream_power.extend(downstream_power_46[i])

# for i in range(0,len(upstream_power_226)):

#     print('pump action yoghurt rifle')
#     upstream_power.extend(upstream_power_226[i])
#     downstream_power.extend(downstream_power_226[i])


# upstream_power.extend(upstream_power_106)
# upstream_power.extend(upstream_power_226)
# upstream_power.extend(upstream_power_46)

# downstream_power.extend(downstream_power_106)
# downstream_power.extend(downstream_power_226)
# downstream_power.extend(downstream_power_46)

print('')
# print(upstream_power_106[0])


upstream_power = np.array(upstream_power)
downstream_power = np.array(downstream_power)


upstream_power = []
downstream_power = []


plt.figure(500)


for i in range(0,len(lists_46)):
        plt.scatter(lists_46[i][4], lists_46[i][5], color='orangered', s=1,label='46 Degrees Data')
        upstream_power.extend(lists_46[i][4])
        downstream_power.extend(lists_46[i][5])
        

for i in range(0,len(lists_226)):
        plt.scatter(lists_226[i][4], lists_226[i][5], color='cornflowerblue', s=1,label='226 Degrees Data')
        upstream_power.extend(lists_226[i][4])
        downstream_power.extend(lists_226[i][5])

for i in range(0,len(lists_106)):
        plt.scatter(lists_106[i][4], lists_106[i][5], color='darkmagenta', s=1,label='106 Degrees Data')
        upstream_power.extend(lists_106[i][4])
        downstream_power.extend(lists_106[i][5])


# plt.plot(x,jensen_speed,color='red', linewidth=4,label="Jensen Fit")
# plt.plot(x,single_wake_data,color='red', linewidth=4,label="Jensen Fit")
plt.xlabel("Upstream Wind Turbine Power (kW)")
plt.ylabel("Downstream Wind Turbine power (kW)")
plt.title("Upstream Vs Downstream Power for turbine in 2 wakes")
plt.legend(loc="upper left")

plt.grid()


plt.figure(5000)


plt.scatter(upstream_power,downstream_power,s=0.5,label="Turbine Power Data")
plt.plot(jensen_x_axis,jensen_y_axis,color='black', linewidth=4,label="Jensen Power")
plt.xlabel("Upstream Wind Turbine Power (kW)")
plt.ylabel("Downstream Wind Turbine power (kW)")
plt.title("Upstream Vs Downstream Power for turbine in 2 wakes")
plt.legend(loc="upper left")

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



# power up vs down, with jensen predicted value as as line graph
#do this for each turbine