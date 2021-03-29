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
df_curtailed = df_curtailed['O09_Grd_Prod_Pwr_InternalDerateStat']

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

angle_lower = 218
angle_higher = 222

power_df=final_df.loc[(final_df['O09_Grd_Prod_Pwr_InternalDerateStat']<4) & (final_df['O09_Amb_WindDir_Abs_Avg']>=angle_lower) & (final_df['O09_Amb_WindDir_Abs_Avg']< angle_higher)][['O09_Amb_WindDir_Abs_Avg','O09_Amb_WindSpeed_Avg', 'WindSpeed_Mean','O09_Grd_Prod_Pwr_Avg']].copy()
speed_df = final_df.loc[(final_df['O09_Grd_Prod_Pwr_InternalDerateStat']<4) & (final_df['O09_Amb_WindDir_Abs_Avg']>=angle_lower) & (final_df['O09_Amb_WindDir_Abs_Avg']< angle_higher)][['O09_Amb_WindSpeed_Avg', 'N09_Amb_WindSpeed_Avg']].copy()

x = np.linspace(0,25,26)
y = np.linspace(0,25,26)

# x1_index=min(range(len(X[0])), key=lambda i: abs(X[0][i]-x1_coord))  # this indexes the recorded power to the nearest powercurve windspeed (correction)
# min(range(len(power_smooth)), key=lambda i: abs([0][i]-x1_coord))

# print(power_df)

#     For each power value:
#   If power < 3450:
#      Map this to a wind speed on the power curve
#   If power = 3450:
#      If if corresponding windspeed > 20:
#         Take the corresponding measured wind speed
corrected_windSpeed = []
for row in power_df.itertuples():

    if row.O09_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.O09_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        if correct_windspeed < 3:
            corrected_windSpeed.append(0)

    if row.O09_Grd_Prod_Pwr_Avg < 3450:
        index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-row.O09_Grd_Prod_Pwr_Avg))
        correct_windspeed = powercurve_windspeed_new[index_speed]
        corrected_windSpeed.append(correct_windspeed)
    if row.O09_Grd_Prod_Pwr_Avg >= 3450:
        correct_windspeed = row.O09_Amb_WindSpeed_Avg
        corrected_windSpeed.append(correct_windspeed)
    

power_df["corrected_windspeed"] = corrected_windSpeed


#overall goal, power to wind speed, to see if the relevant wind speeds come out as the same power seen by jensens
#1) get actual wind speed on lead turbine. power to corresponding wind speed (corrected wind speed) 
#2) apply jensens to get next downstream wind speed
#3) work out power of next downstream turbine via power curve
#4) compare this predicted jensen power to the actual power 


# windspeeds = np.linspace(0,25,26)
# power_generation_kw = []
# for i in range(0,len(windspeeds)):
#     wind_speed = windspeeds[i]
#     if wind_speed < 3 or wind_speed > 25: #cut in and cut out
#         power = 0
#     elif wind_speed >= 3 and wind_speed <= 12.5: #power curve
#         power = 0.0676*wind_speed**6 - 3.2433*wind_speed**5 + 60.607*wind_speed**4 - 565.82*wind_speed**3 + 2830.8*wind_speed**2 - 7083.5*wind_speed + 6896.3
#     elif wind_speed > 12.5 and wind_speed <= 25: # max power
#         power = 3450
#     power_generation_kw.append(power)


Ct = 0.8
rd=23.5
kw = 0.04
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
plt.scatter(power_df['corrected_windspeed'],power_df['O09_Grd_Prod_Pwr_Avg'],marker='x',color='green',s=10,label='Corrected Wind Speed Data')
plt.plot(powercurve_windspeed_new,power_smooth,color='orange',label='Power Curve')
plt.legend(loc="upper left")



plt.figure(2)
plt.xlabel('Wind Speed Upstream Turbine m/s ', fontsize=12)
plt.xticks(fontsize= 12)
plt.ylabel('Wind Speed Downstream Turbine m/s', fontsize=14)
plt.title('Wind Turbine Speed Deficiency')
plt.grid()
plt.scatter(speed_df['O09_Amb_WindSpeed_Avg'],speed_df['N09_Amb_WindSpeed_Avg'],marker='x',s=1,label='Turbie Speed')
plt.plot(x,jensen_speed,color='orange',label='Jensen')
plt.plot(x,y,color='green',label='No Deficit')
plt.legend(loc="upper left")

plt.show()



#lead wind turbine speed on x axis
#y axis speed on turbine behind


#print(df.loc[(df['WindSpeed_Mean']>=0) & (df['WindSpeed_Mean']< 5)])

#print(new_df.loc[(new_df['O09_Amb_WindDir_Abs_Avg']>=0) & (new_df['O09_Amb_WindDir_Abs_Avg']< 5)])

