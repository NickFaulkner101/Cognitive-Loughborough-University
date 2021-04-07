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
# Note at the moment it just filters when O09 is curtailed.
df_curtailed = pd.read_csv("curtailed_setting.csv")
df_curtailed.index=df_curtailed['timestamp']
df_curtailed = df_curtailed.drop('timestamp', axis =1)
# df_curtailed = df_curtailed['O09_Grd_Prod_Pwr_InternalDerateStat']

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

#Taking bottom left wind turbine, 'O09'

#Taking bottom left wind turbine, 'I15'
#46 degree bearing, 226 wind dir
angle_lower = 224
angle_higher = 228

power_df=final_df.loc[
 
 (final_df['O09_Grd_Prod_Pwr_Avg'] > 0) &   
 (final_df['N09_Grd_Prod_Pwr_Avg'] > 0) & #this removes null values that pandas struggles with
 (final_df['M09_Grd_Prod_Pwr_Avg'] > 0) &   
 (final_df['O09_Grd_Prod_Pwr_InternalDerateStat']<4) & #this removes curtailed values
 (final_df['N09_Grd_Prod_Pwr_InternalDerateStat']<4) &
 (final_df['M09_Grd_Prod_Pwr_InternalDerateStat']<4) &
 (final_df['O09_Amb_WindDir_Abs_Avg']>=angle_lower) & #220 degrees as the turbines of interest are aligned along this plane for wind dir
 (final_df['O09_Amb_WindDir_Abs_Avg']< angle_higher)][[
     'O09_Amb_WindDir_Abs_Avg','O09_Amb_WindSpeed_Avg',
     'N09_Amb_WindSpeed_Avg',
     'M09_Amb_WindSpeed_Avg', 
     'WindSpeed_Mean',
     'O09_Grd_Prod_Pwr_Avg',
     'N09_Grd_Prod_Pwr_Avg',
     'M09_Grd_Prod_Pwr_Avg']].copy()

speed_df = final_df.loc[(final_df['N09_Grd_Prod_Pwr_Avg'] > 0) & 
(final_df['O09_Grd_Prod_Pwr_InternalDerateStat']<4) & 
(final_df['O09_Amb_WindDir_Abs_Avg']>=angle_lower) & 
(final_df['O09_Amb_WindDir_Abs_Avg']< angle_higher)][['O09_Amb_WindSpeed_Avg', 'N09_Amb_WindSpeed_Avg']].copy()

#Jensens Variables
Ct = 0.8
rd=56
kw = 0.04


x = np.linspace(0,25,26)
y = np.linspace(0,25,26)

# x1_index=min(range(len(X[0])), key=lambda i: abs(X[0][i]-x1_coord))  # this indexes the recorded power to the nearest powercurve windspeed (correction)
# min(range(len(power_smooth)), key=lambda i: abs([0][i]-x1_coord))

print('power_df length'+ str(len(power_df)))

#O09 windspeed correction via powercurve 
corrected_O09_windspeed = []     
for row in power_df.itertuples():

    if row.O09_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.O09_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        if correct_windspeed < 3:
            corrected_O09_windspeed.append(0)

    if row.O09_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.O09_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        corrected_O09_windspeed.append(correct_windspeed)
    if row.O09_Grd_Prod_Pwr_Avg >= 3450:
        correct_windspeed = row.O09_Amb_WindSpeed_Avg
        corrected_O09_windspeed.append(correct_windspeed)

print(len(corrected_O09_windspeed))
power_df["corrected_O09_windspeed"] = corrected_O09_windspeed

#N09 windspeed correction via powercurve 
corrected_N09_windspeed = []     
for row in power_df.itertuples():

    if row.N09_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.N09_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        if correct_windspeed < 3:
            corrected_N09_windspeed.append(0)

    if row.N09_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.N09_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        corrected_N09_windspeed.append(correct_windspeed)
    if row.N09_Grd_Prod_Pwr_Avg >= 3450:
        correct_windspeed = row.N09_Amb_WindSpeed_Avg
        corrected_N09_windspeed.append(correct_windspeed)

power_df["corrected_N09_windspeed"] = corrected_N09_windspeed


corrected_M09_windspeed = []     
for row in power_df.itertuples():

    if row.M09_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.M09_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        if correct_windspeed < 3:
            corrected_M09_windspeed.append(0)

    if row.M09_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.M09_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        corrected_M09_windspeed.append(correct_windspeed)
    if row.M09_Grd_Prod_Pwr_Avg >= 3450:
        correct_windspeed = row.M09_Amb_WindSpeed_Avg
        corrected_M09_windspeed.append(correct_windspeed)

power_df["corrected_M09_windspeed"] = corrected_M09_windspeed


print(len(corrected_N09_windspeed))
print(len(corrected_O09_windspeed))





#N09 Jensen windspeed from O09 hence power prediction 
N09_windspeed_Jensen= []
N09_power = []

turbine_distance = 778.7 #distance from O09 to N09 in metres
for row in power_df.itertuples():
    O09_wind = row.corrected_O09_windspeed

    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    N09_wind = factor*O09_wind
    N09_windspeed_Jensen.append(N09_wind)

    index_power = min(range(len(powercurve_windspeed_new)), key=lambda i: abs(powercurve_windspeed_new[i]-N09_wind))
    N09_power_value = power_smooth[index_power]
    N09_power.append(N09_power_value)
  
power_df["N09_power_Jensen"] = N09_power
power_df["N09_windspeed_Jensen"] = N09_windspeed_Jensen

#M09 Jensen windspeed from N09 
M09_windspeed_Jensen= []
M09_power = []

turbine_distance = 788.1 #distance from N09 to M09 in metres
for row in power_df.itertuples():
    N09_wind = row.corrected_N09_windspeed

    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    M09_wind = factor*N09_wind
    M09_windspeed_Jensen.append(factor)

    index_power = min(range(len(powercurve_windspeed_new)), key=lambda i: abs(powercurve_windspeed_new[i]-M09_wind))
    M09_power_value = power_smooth[index_power]
    M09_power.append(M09_power_value)
  
power_df["M09_N09_Jensen_factor"] = M09_windspeed_Jensen



#M09 Jensen windspeed from O09 
M09_windspeed_Jensen= []

turbine_distance = 788.1 #distance from O09 to M09 in metres
for row in power_df.itertuples():
    O09_wind = row.corrected_O09_windspeed

    factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
    M09_wind = factor*O09_wind
    M09_windspeed_Jensen.append(factor)

    
  
power_df["M09_O09_Jensen_factor"] = M09_windspeed_Jensen

#calculates predicted Jensen Speed on M09 via RSS Method
Final_Jensen_Speed_M09 = []
M09_power = []
for row in power_df.itertuples():
    O09_factor = row.M09_O09_Jensen_factor
    N09_factor = row.M09_N09_Jensen_factor
    O09_wind = row.corrected_O09_windspeed


    _M09_factor = np.sqrt(np.square(1 - O09_factor)+np.square(1 - N09_factor))  #ROOT SUM OF SQUARES
    M09_factor = 1 - _M09_factor
    actual_speed = M09_factor*O09_wind
    Final_Jensen_Speed_M09.append(actual_speed) #multiply by freestream velocity

    index_power = min(range(len(powercurve_windspeed_new)), key=lambda i: abs(powercurve_windspeed_new[i]-actual_speed))
    M09_power_value = power_smooth[index_power]
    M09_power.append(M09_power_value)


  
power_df["M09_Final_Jensen_Speed"] = Final_Jensen_Speed_M09
  
power_df["M09_power_Jensen"] = M09_power
print(power_df["M09_Final_Jensen_Speed"])




turbine_distance = 778.7
jensen_speed = []
factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
for i in range(0,len(x)):
    
    speed_deficit = factor*x[i]
    jensen_speed.append(speed_deficit)



plt.figure(1)
plt.xlabel('Wind Speed m/s ', fontsize=12)
plt.xticks(fontsize= 12)
plt.ylabel('power', fontsize=14)
plt.title('Wind Speed Distribution')
plt.grid()
plt.scatter(power_df['O09_Amb_WindSpeed_Avg'],power_df['O09_Grd_Prod_Pwr_Avg'],marker='x',s=1,label='Turbine Data')
plt.scatter(power_df['corrected_O09_windspeed'],power_df['O09_Grd_Prod_Pwr_Avg'],marker='x',color='green',s=10,label='Corrected Wind Speed Data')
plt.plot(powercurve_windspeed_new,power_smooth,color='orange',label='Power Curve')
plt.legend(loc="upper left")



plt.figure(2)
plt.xlabel('Wind Speed O09 Raw Turbine m/s ', fontsize=12)
plt.xticks(fontsize= 12)
plt.ylabel('Wind Speed N09 Turbine m/s', fontsize=14)
plt.title('Wind Turbine Speed Deficiency')
plt.grid()
plt.scatter(speed_df['O09_Amb_WindSpeed_Avg'],speed_df['N09_Amb_WindSpeed_Avg'],marker='x',s=1,label='Turbie Speed')
plt.plot(x,jensen_speed,color='orange',label='Jensen')
plt.plot(x,y,color='green',label='No Deficit')
plt.legend(loc="upper left")

plt.figure(4)
plt.title('Comparison of N09 Power Values and Jensen-Based Power')
plt.xlabel('Wind Speed m/s ', fontsize=12)
plt.xticks(fontsize= 12)
plt.ylabel('power /kW', fontsize=14)
plt.grid()
plt.scatter(power_df['corrected_N09_windspeed'],power_df['N09_Grd_Prod_Pwr_Avg'],marker='x',s=1,label='Turbine Data')
plt.scatter(power_df['N09_windspeed_Jensen'],power_df['N09_power_Jensen'],marker='x',s=5,color='green',label='Jensen Power')
plt.plot(powercurve_windspeed_new,power_smooth,color='orange',label='Power Curve')
plt.legend(loc="upper left")


plt.figure(5)
plt.xlabel('Wind Speed O09 Turbine m/s ', fontsize=12)
plt.xticks(fontsize= 12)
plt.ylabel('Wind Speed N09 Turbine m/s', fontsize=14)
plt.title('Wind Turbine Wind Speed Difference')
plt.grid()

plt.scatter(power_df["corrected_O09_windspeed"],power_df["N09_windspeed_Jensen"],
marker='x',s=5, label='Jensen Windspeed Prediction')

plt.scatter(power_df["corrected_O09_windspeed"],power_df["corrected_N09_windspeed"],
marker='x',s=5,color='green',label='(Corrected) Turbine Windspeed')
plt.legend(loc="upper left")


plt.figure(6)
plt.xlabel('Upwind Turbine Power Output /kW', fontsize=14)
plt.xticks(fontsize= 12)
plt.ylabel('N09 Turbine Power Output kW', fontsize=14)
plt.title('Power Relationship Between O09 and N09')
plt.grid()

plt.scatter(power_df["O09_Grd_Prod_Pwr_Avg"],power_df["N09_Grd_Prod_Pwr_Avg"],
marker='x',s=5, label='Actual Power Relationship')

plt.scatter(power_df["O09_Grd_Prod_Pwr_Avg"],power_df["N09_power_Jensen"],
marker='x',s=5,color='green',label='Jensen Predicted Power Relationship')
plt.legend(loc="upper left")



plt.figure(7)
plt.xlabel('Upwind Turbine Power Output /kW', fontsize=14)
plt.xticks(fontsize= 12)
plt.ylabel('N09 Turbine Power Output kW', fontsize=14)
plt.title('Power Relationship Between O09 and M09')
plt.grid()

plt.scatter(power_df["O09_Grd_Prod_Pwr_Avg"],power_df["M09_Grd_Prod_Pwr_Avg"],
marker='x',s=5,color='red', label='O09 and M09')
plt.scatter(power_df["O09_Grd_Prod_Pwr_Avg"],power_df["N09_Grd_Prod_Pwr_Avg"],
marker='x',s=5,color='green', label='O09 and N09')



plt.figure(8)
plt.xlabel('Wind Speed O09 Turbine m/s ', fontsize=12)
plt.xticks(fontsize= 12)
plt.ylabel('Wind Speed M09 Turbine m/s', fontsize=14)
plt.title('Two Wakes Jensen Model')
plt.grid()

plt.scatter(power_df["corrected_O09_windspeed"],power_df["M09_Final_Jensen_Speed"],
marker='x',s=5, label='Jensen Windspeed Prediction')

plt.scatter(power_df["corrected_O09_windspeed"],power_df["corrected_M09_windspeed"],
marker='x',s=5,color='green',label='(Corrected) Turbine Windspeed')
plt.legend(loc="upper left")

plt.figure(9)
plt.xlabel('Upwind Turbine Power Output /kW', fontsize=14)
plt.xticks(fontsize= 12)
plt.ylabel('M09 Turbine Power Output kW', fontsize=14)
plt.title('Power Relationship Between O09 and M09 (M09 in 2 Wakes)')
plt.grid()

plt.scatter(power_df["O09_Grd_Prod_Pwr_Avg"],power_df["M09_Grd_Prod_Pwr_Avg"],
marker='x',s=5, label='Actual Power Relationship')

plt.scatter(power_df["O09_Grd_Prod_Pwr_Avg"],power_df["M09_power_Jensen"],
marker='x',s=5,color='green',label='Jensen Predicted Power Relationship')
plt.legend(loc="upper left")







#power O09, vs jensen predicted N09 in 2 wakes
#so,
# 1) get M09 jensen factors from O09, 
# 2) get M09 jensen factors from N09
#    a) use least squares to find actual jensen factor at the point
#    b) find speed

# plt.scatter(power_df["O09_Grd_Prod_Pwr_Avg"],power_df["N09_power_Jensen"],
# marker='x',s=5,color='green',label='Jensen Predicted Power Relationship')
plt.legend(loc="upper left")




#compare N09 T2 corrected windspeed measurement (p2 ----> w2) to (p1 ---> w1)  w1 vs w2 
# compared against w1 vs jensen2

#compare N09 power produced for a given windspeed







plt.show()





#what am I comparing
#I want to see that for a given O09 wind speed, the wind speeds for jensens prediction are the same as reality
#and, the power output for N09 is predicted the same 
# ransac ?




#lead wind turbine speed on x axis
#y axis speed on turbine behind


#print(df.loc[(df['WindSpeed_Mean']>=0) & (df['WindSpeed_Mean']< 5)])

#print(new_df.loc[(new_df['O09_Amb_WindDir_Abs_Avg']>=0) & (new_df['O09_Amb_WindDir_Abs_Avg']< 5)])

