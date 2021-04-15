import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import make_interp_spline, BSpline


# Load in WindSpeed Data
df = pd.read_csv("WindSpeed_Average.csv")
df.index=df['timestamp']
df = df.drop('timestamp', axis =1)
df['WindSpeed_Mean'] = df.mean(axis=1)

#Load in Wind Dir Data
df_Dir = pd.read_csv("WindDir_Data.csv")
df_Dir.index=df_Dir['timestamp']
df_Dir = df_Dir.drop('timestamp', axis =1)

#Load in Wind Power Data
df_power = pd.read_csv("power.csv")
df_power.index=df_power['timestamp']
df_power = df_power.drop('timestamp', axis =1)

#Load In Curtailed Data
# Note at the moment it just filters when N10 is curtailed.
df_curtailed = pd.read_csv("curtailed_setting.csv")
df_curtailed.index=df_curtailed['timestamp']
df_curtailed = df_curtailed.drop('timestamp', axis =1)
# df_curtailed = df_curtailed['N10_Grd_Prod_Pwr_InternalDerateStat']

#power Curve load in 
# Note that this is for the VESTAS v112 taken from a 3rd party website. 
# This splices existing data points into a few hundred more points
powercurve_windspeed= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5])
powercurve_windspeed_new = np.linspace(powercurve_windspeed.min(), powercurve_windspeed.max(), 3000)  
powercurve_power = [7,53,123,208,309,427,567,732,927,1149,1401,1688,2006,2348,2693,3011,3252,3388,3436,3448,3450]

spl = make_interp_spline(powercurve_windspeed, powercurve_power, k=1)  # type: BSpline
#powercurve load in
power_smooth = spl(powercurve_windspeed_new)
plt.figure(3)
plt.plot(powercurve_windspeed_new, power_smooth,label='New')
plt.legend(loc="upper left")





#Merge Dataframes
new_df=df.merge(df_Dir,left_index=True,right_index=True)
new_new_df=new_df.merge(df_curtailed,left_index=True,right_index=True)
final_df=new_new_df.merge(df_power,left_index=True,right_index=True)

#Taking bottom left wind turbine, 'N10'

#Taking bottom left wind turbine, 'I15'
#46 degree bearing, 226 wind dir
angle_lower = 224
angle_higher = 228

power_df=final_df.loc[
 
 (final_df['N10_Grd_Prod_Pwr_Avg'] > 0) &   
 (final_df['M10_Grd_Prod_Pwr_Avg'] > 0) & #this removes null values that pandas struggles with
 (final_df['L10_Grd_Prod_Pwr_Avg'] > 0) & 
 (final_df['K10_Grd_Prod_Pwr_Avg'] > 0) & 
 (final_df['J10_Grd_Prod_Pwr_Avg'] > 0) & 
 (final_df['N10_Grd_Prod_Pwr_InternalDerateStat']<4) & #this removes curtailed values
 (final_df['M10_Grd_Prod_Pwr_InternalDerateStat']<4) &
 (final_df['L10_Grd_Prod_Pwr_InternalDerateStat']<4) &
 (final_df['K10_Grd_Prod_Pwr_InternalDerateStat']<4) &
 (final_df['J10_Grd_Prod_Pwr_InternalDerateStat']<4) &
 (final_df['N10_Amb_WindDir_Abs_Avg']>=angle_lower) & #220 degrees as the turbines of interest are aligned along this plane for wind dir
 (final_df['N10_Amb_WindDir_Abs_Avg']< angle_higher)][[
     'N10_Amb_WindDir_Abs_Avg',
     'N10_Amb_WindSpeed_Avg',
     'M10_Amb_WindSpeed_Avg',
     'L10_Amb_WindSpeed_Avg',
     'K10_Amb_WindSpeed_Avg', 
     'J10_Amb_WindSpeed_Avg', 
     'WindSpeed_Mean',
     'N10_Grd_Prod_Pwr_Avg',
     'M10_Grd_Prod_Pwr_Avg',
     'L10_Grd_Prod_Pwr_Avg',
     'K10_Grd_Prod_Pwr_Avg',
     'J10_Grd_Prod_Pwr_Avg']].copy()

speed_df = final_df.loc[(final_df['M10_Grd_Prod_Pwr_Avg'] > 0) & 
(final_df['N10_Grd_Prod_Pwr_InternalDerateStat']<4) & 
(final_df['N10_Amb_WindDir_Abs_Avg']>=angle_lower) & 
(final_df['N10_Amb_WindDir_Abs_Avg']< angle_higher)][['N10_Amb_WindSpeed_Avg', 'M10_Amb_WindSpeed_Avg']].copy()

#Jensens Variables
Ct = 0.8
rd=56
kw = 0.04


x = np.linspace(0,25,26)
y = np.linspace(0,25,26)

# x1_index=min(range(len(X[0])), key=lambda i: abs(X[0][i]-x1_coord))  # this indexes the recorded power to the nearest powercurve windspeed (correction)
# min(range(len(power_smooth)), key=lambda i: abs([0][i]-x1_coord))

print('power_df length'+ str(len(power_df)))

#N10 windspeed correction via powercurve 
corrected_N10_windspeed = []     
for row in power_df.itertuples():

    if row.N10_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.N10_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        if correct_windspeed < 3:
            corrected_N10_windspeed.append(0)

    if row.N10_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.N10_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        corrected_N10_windspeed.append(correct_windspeed)
    if row.N10_Grd_Prod_Pwr_Avg >= 3450:
        correct_windspeed = row.N10_Amb_WindSpeed_Avg
        corrected_N10_windspeed.append(correct_windspeed)

print(len(corrected_N10_windspeed))
power_df["corrected_N10_windspeed"] = corrected_N10_windspeed

#M10 windspeed correction via powercurve 
corrected_M10_windspeed = []     
for row in power_df.itertuples():

    if row.M10_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.M10_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        if correct_windspeed < 3:
            corrected_M10_windspeed.append(0)

    if row.M10_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.M10_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        corrected_M10_windspeed.append(correct_windspeed)
    if row.M10_Grd_Prod_Pwr_Avg >= 3450:
        correct_windspeed = row.M10_Amb_WindSpeed_Avg
        corrected_M10_windspeed.append(correct_windspeed)

power_df["corrected_M10_windspeed"] = corrected_M10_windspeed


corrected_L10_windspeed = []     
for row in power_df.itertuples():

    if row.L10_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.L10_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        if correct_windspeed < 3:
            corrected_L10_windspeed.append(0)

    if row.L10_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.L10_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        corrected_L10_windspeed.append(correct_windspeed)
    if row.L10_Grd_Prod_Pwr_Avg >= 3450:
        correct_windspeed = row.L10_Amb_WindSpeed_Avg
        corrected_L10_windspeed.append(correct_windspeed)

power_df["corrected_L10_windspeed"] = corrected_L10_windspeed

corrected_K10_windspeed = []     
for row in power_df.itertuples():

    if row.K10_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.K10_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        if correct_windspeed < 3:
            corrected_K10_windspeed.append(0)

    if row.K10_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.K10_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        corrected_K10_windspeed.append(correct_windspeed)
    if row.K10_Grd_Prod_Pwr_Avg >= 3450:
        correct_windspeed = row.K10_Amb_WindSpeed_Avg
        corrected_K10_windspeed.append(correct_windspeed)

power_df["corrected_K10_windspeed"] = corrected_K10_windspeed


print(len(corrected_M10_windspeed))
print(len(corrected_N10_windspeed))





#M10 Jensen windspeed from N10 hence power prediction 
M10_windspeed_Jensen= []
M10_power = []

turbine_distance = 778.7 #distance from N10 to M10 in metres
for row in power_df.itertuples():
    N10_wind = row.corrected_N10_windspeed

    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    M10_wind = factor*N10_wind
    M10_windspeed_Jensen.append(M10_wind)

    index_power = min(range(len(powercurve_windspeed_new)), key=lambda i: abs(powercurve_windspeed_new[i]-M10_wind))
    M10_power_value = power_smooth[index_power]
    M10_power.append(M10_power_value)
  
power_df["M10_power_Jensen"] = M10_power
power_df["M10_windspeed_Jensen"] = M10_windspeed_Jensen

#Jensen windspeed from M10 
L10_windspeed_Jensen= []
K10_windspeed_Jensen= []

for row in power_df.itertuples():
    turbine_distance = 788.1 #distance from M10 to L10 in metres
    M10_wind = row.corrected_M10_windspeed

    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    L10_wind = factor*M10_wind
    L10_windspeed_Jensen.append(factor)

    turbine_distance = 1573 #distance from M10 to K10 in metres

    M10_wind = row.corrected_M10_windspeed

    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    K10_wind = factor*M10_wind
    K10_windspeed_Jensen.append(factor)
  
power_df["L10_M10_Jensen_factor"] = L10_windspeed_Jensen
power_df["K10_M10_Jensen_factor"] = K10_windspeed_Jensen



#Jensen windspeeds from N10  
L10_windspeed_Jensen= []
K10_windspeed_Jensen= []
for row in power_df.itertuples():
    turbine_distance = 1567 #distance from N10 to L10 in metres

    N10_wind = row.corrected_N10_windspeed

    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    L10_wind = factor*N10_wind
    L10_windspeed_Jensen.append(factor)

    turbine_distance = 2352 #distance from N10 to K10 in metres

    N10_wind = row.corrected_N10_windspeed

    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    K10_wind = factor*N10_wind
    K10_windspeed_Jensen.append(factor)

  
power_df["L10_N10_Jensen_factor"] = L10_windspeed_Jensen
power_df["K10_N10_Jensen_factor"] = K10_windspeed_Jensen

#calculate jensen factor for K10 from L10
K10_windspeed_Jensen= []
for row in power_df.itertuples():
    turbine_distance = 1576 #distance from L10 to K10 in metres
    N10_wind = row.corrected_N10_windspeed
    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    K10_wind = factor*N10_wind
    K10_windspeed_Jensen.append(factor)

  
power_df["K10_L10_Jensen_factor"] = K10_windspeed_Jensen



#calculates predicted Jensen Speed on L10 via RSS Method
Final_Jensen_Speed_L10 = []
L10_power = []
for row in power_df.itertuples():
    N10_factor = row.L10_N10_Jensen_factor
    M10_factor = row.L10_M10_Jensen_factor
    N10_wind = row.corrected_N10_windspeed


    _L10_factor = np.sqrt(np.square(1 - N10_factor)+np.square(1 - M10_factor))  #ROOT SUM OF SQUARES
    L10_factor = 1 - _L10_factor
    actual_speed = L10_factor*N10_wind
    Final_Jensen_Speed_L10.append(actual_speed) #multiply by freestream velocity

    index_power = min(range(len(powercurve_windspeed_new)), key=lambda i: abs(powercurve_windspeed_new[i]-actual_speed))
    L10_power_value = power_smooth[index_power]
    L10_power.append(L10_power_value)


  
power_df["L10_Final_Jensen_Speed"] = Final_Jensen_Speed_L10
  
power_df["L10_power_Jensen"] = L10_power
print(power_df["L10_Final_Jensen_Speed"])

#calculates predicted Jensen Speed on K10 via RSS Method
Final_Jensen_Speed_K10 = []
K10_power = []
for row in power_df.itertuples():
    N10_factor = row.K10_N10_Jensen_factor
    M10_factor = row.K10_M10_Jensen_factor
    L10_factor = row.K10_L10_Jensen_factor
    N10_wind = row.corrected_N10_windspeed

    previous_jensen = np.sqrt(np.square(1 - N10_factor)+np.square(1 - M10_factor)) #lies in 3 wakes, N10, M10 and L10
    _K10_factor = np.sqrt(np.square(1 - L10_factor)+np.square(previous_jensen))  #ROOT SUM OF SQUARES
    K10_factor = 1 - _K10_factor
    actual_speed = K10_factor*N10_wind
    Final_Jensen_Speed_K10.append(actual_speed) #multiply by freestream velocity

    index_power = min(range(len(powercurve_windspeed_new)), key=lambda i: abs(powercurve_windspeed_new[i]-actual_speed))
    K10_power_value = power_smooth[index_power]
    K10_power.append(K10_power_value)


  
power_df["K10_Final_Jensen_Speed"] = Final_Jensen_Speed_K10
  
power_df["K10_power_Jensen"] = K10_power


turbine_distance = 778.7
jensen_speed = []
factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
for i in range(0,len(x)):
    
    speed_deficit = factor*x[i]
    jensen_speed.append(speed_deficit)



# plt.figure(1)
# plt.xlabel('Wind Speed m/s ', fontsize=12)
# plt.xticks(fontsize= 12)
# plt.ylabel('power', fontsize=14)
# plt.title('Wind Speed Distribution')
# plt.grid()
# plt.scatter(power_df['N10_Amb_WindSpeed_Avg'],power_df['N10_Grd_Prod_Pwr_Avg'],marker='x',s=1,label='Turbine Data')
# plt.scatter(power_df['corrected_N10_windspeed'],power_df['N10_Grd_Prod_Pwr_Avg'],marker='x',color='green',s=10,label='Corrected Wind Speed Data')
# plt.plot(powercurve_windspeed_new,power_smooth,color='orange',label='Power Curve')
# plt.legend(loc="upper left")



# plt.figure(2)
# plt.xlabel('Wind Speed N10 Raw Turbine m/s ', fontsize=12)
# plt.xticks(fontsize= 12)
# plt.ylabel('Wind Speed M10 Turbine m/s', fontsize=14)
# plt.title('Wind Turbine Speed Deficiency')
# plt.grid()
# plt.scatter(speed_df['N10_Amb_WindSpeed_Avg'],speed_df['M10_Amb_WindSpeed_Avg'],marker='x',s=1,label='Turbie Speed')
# plt.plot(x,jensen_speed,color='orange',label='Jensen')
# plt.plot(x,y,color='green',label='No Deficit')
# plt.legend(loc="upper left")

# plt.figure(4)
# plt.title('Comparison of M10 Power Values and Jensen-Based Power')
# plt.xlabel('Wind Speed m/s ', fontsize=12)
# plt.xticks(fontsize= 12)
# plt.ylabel('power /kW', fontsize=14)
# plt.grid()
# plt.scatter(power_df['corrected_M10_windspeed'],power_df['M10_Grd_Prod_Pwr_Avg'],marker='x',s=1,label='Turbine Data')
# plt.scatter(power_df['M10_windspeed_Jensen'],power_df['M10_power_Jensen'],marker='x',s=5,color='green',label='Jensen Power')
# plt.plot(powercurve_windspeed_new,power_smooth,color='orange',label='Power Curve')
# plt.legend(loc="upper left")

plt.figure(40)
plt.title('N10 vs M10 WindSpeed (Single Wake')
plt.xlabel('Wind Speed m/s ', fontsize=12)
plt.xticks(fontsize= 12)
plt.ylabel('power /kW', fontsize=14)
plt.grid()
plt.scatter(power_df['corrected_N10_windspeed'],power_df['corrected_M10_windspeed'],marker='x',s=1,label='Turbine Data')
plt.scatter(power_df['corrected_N10_windspeed'],power_df['M10_windspeed_Jensen'],marker='x',s=5,color='green',label='Jensen Power')
plt.legend(loc="upper left")


plt.figure(41)
plt.title('N10 vs L10 WindSpeed (Two Wakes)')
plt.xlabel('Wind Speed m/s ', fontsize=12)
plt.xticks(fontsize= 12)
plt.ylabel('power /kW', fontsize=14)
plt.grid()
plt.scatter(power_df['corrected_N10_windspeed'],power_df['corrected_L10_windspeed'],marker='x',s=1,label='Turbine Data')
plt.scatter(power_df['corrected_N10_windspeed'],power_df['L10_Final_Jensen_Speed'],marker='x',s=5,color='green',label='Jensen Power')
plt.legend(loc="upper left")

plt.figure(42)
plt.title('N10 vs K10 WindSpeed (Three Wakes)')
plt.xlabel('Wind Speed m/s ', fontsize=12)
plt.xticks(fontsize= 12)
plt.ylabel('power /kW', fontsize=14)
plt.grid()
plt.scatter(power_df['corrected_N10_windspeed'],power_df['corrected_K10_windspeed'],marker='x',s=1,label='Turbine Data')
plt.scatter(power_df['corrected_N10_windspeed'],power_df['K10_Final_Jensen_Speed'],marker='x',s=5,color='green',label='Jensen Power')
plt.legend(loc="upper left")

plt.figure(43)
plt.xlabel('Upwind Turbine Power Output /kW', fontsize=14)
plt.xticks(fontsize= 12)
plt.ylabel('L10 Turbine Power Output kW', fontsize=14)
plt.title('Power Relationship Between N10 and L10 (L10 in 2 Wakes)')
plt.grid()

plt.scatter(power_df["N10_Grd_Prod_Pwr_Avg"],power_df["L10_Grd_Prod_Pwr_Avg"],
marker='x',s=5, label='Actual Power Relationship')

plt.scatter(power_df["N10_Grd_Prod_Pwr_Avg"],power_df["L10_power_Jensen"],
marker='x',s=5,color='green',label='Jensen Predicted Power Relationship')
plt.legend(loc="upper left")

plt.figure(44)
plt.xlabel('Upwind Turbine Power Output /kW', fontsize=14)
plt.xticks(fontsize= 12)
plt.ylabel('M10 Turbine Power Output kW', fontsize=14)
plt.title('Power Relationship Between N10 and M10')
plt.grid()

plt.scatter(power_df["N10_Grd_Prod_Pwr_Avg"],power_df["M10_Grd_Prod_Pwr_Avg"],
marker='x',s=5, label='Actual Power Relationship')

plt.scatter(power_df["N10_Grd_Prod_Pwr_Avg"],power_df["M10_power_Jensen"],
marker='x',s=5,color='green',label='Jensen Predicted Power Relationship')
plt.legend(loc="upper left")

plt.figure(45)
plt.xlabel('Upwind Turbine Power Output /kW', fontsize=14)
plt.xticks(fontsize= 12)
plt.ylabel('K10 Turbine Power Output kW', fontsize=14)
plt.title('Power Relationship Between N10 and K10')
plt.grid()

plt.scatter(power_df["N10_Grd_Prod_Pwr_Avg"],power_df["K10_Grd_Prod_Pwr_Avg"],
marker='x',s=5, label='Actual Power Relationship')

plt.scatter(power_df["N10_Grd_Prod_Pwr_Avg"],power_df["K10_power_Jensen"],
marker='x',s=5,color='green',label='Jensen Predicted Power Relationship')
plt.legend(loc="upper left")






plt.figure(46)
plt.xlabel('Upwind Turbine Power Output /kW', fontsize=14)
plt.xticks(fontsize= 12)
plt.ylabel('M10 Turbine Power Output kW', fontsize=14)
plt.title('Upstream Vs Downstream Turbine Power Output')
plt.grid()


plt.scatter(power_df["N10_Grd_Prod_Pwr_Avg"],power_df["M10_Grd_Prod_Pwr_Avg"],
marker='x',s=5,color='red', label='N10 and M10')

plt.scatter(power_df["N10_Grd_Prod_Pwr_Avg"],power_df["L10_Grd_Prod_Pwr_Avg"],
marker='x',s=5,color='orange', label='N10 and L10')


plt.scatter(power_df["N10_Grd_Prod_Pwr_Avg"],power_df["K10_Grd_Prod_Pwr_Avg"],
marker='x',s=5,color='green', label='N10 and K10')
plt.legend(loc="upper left")











#power N10, vs jensen predicted M10 in 2 wakes
#so,
# 1) get L10 jensen factors from N10, 
# 2) get L10 jensen factors from M10
#    a) use least squares to find actual jensen factor at the point
#    b) find speed

# plt.scatter(power_df["N10_Grd_Prod_Pwr_Avg"],power_df["M10_power_Jensen"],
# marker='x',s=5,color='green',label='Jensen Predicted Power Relationship')
plt.legend(loc="upper left")




#compare M10 T2 corrected windspeed measurement (p2 ----> w2) to (p1 ---> w1)  w1 vs w2 
# compared against w1 vs jensen2

#compare M10 power produced for a given windspeed







plt.show()





#what am I comparing
#I want to see that for a given N10 wind speed, the wind speeds for jensens prediction are the same as reality
#and, the power output for M10 is predicted the same 
# ransac ?




#lead wind turbine speed on x axis
#y axis speed on turbine behind


#print(df.loc[(df['WindSpeed_Mean']>=0) & (df['WindSpeed_Mean']< 5)])

#print(new_df.loc[(new_df['N10_Amb_WindDir_Abs_Avg']>=0) & (new_df['N10_Amb_WindDir_Abs_Avg']< 5)])

