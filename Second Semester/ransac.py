# 1) Understand RANSAC, do a ransac
# 2) compare ransac to the 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import make_interp_spline, BSpline
from sklearn import linear_model, datasets
import sys

def get_ransac(lead,behind):
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

    #power Curve load in 
    # Note that this is for the VESTAS v112 taken from a 3rd party website. 
    # This splices existing data points into a few hundred more points
    powercurve_windspeed= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5])
    powercurve_windspeed_new = np.linspace(powercurve_windspeed.min(), powercurve_windspeed.max(), 3000)  
    powercurve_power = [7,53,123,208,309,427,567,732,927,1149,1401,1688,2006,2348,2693,3011,3252,3388,3436,3448,3450]

    spl = make_interp_spline(powercurve_windspeed, powercurve_power, k=1)  # type: BSpline
    #powercurve load in
    power_smooth = spl(powercurve_windspeed_new)
    # plt.figure(3)
    # plt.plot(powercurve_windspeed_new, power_smooth,label='New')
    # plt.legend(loc="upper left")

    #Merge Dataframes
    new_df=df.merge(df_Dir,left_index=True,right_index=True)
    new_new_df=new_df.merge(df_curtailed,left_index=True,right_index=True)
    final_df=new_new_df.merge(df_power,left_index=True,right_index=True)

    #Taking bottom left wind turbine, 'N10'

    #Taking bottom left wind turbine, 'I15'
    #46 degree bearing, 226 wind dir
    # 106 degrees for h 15 to h 14, 750metre distance
    angle_lower = 224
    angle_higher = 228

    power_df=final_df.loc[
    
    (final_df['N10_Grd_Prod_Pwr_Avg'] > 0) &   
    (final_df['M10_Grd_Prod_Pwr_Avg'] > 0) & #this removes null values that pandas struggles with
    
    (final_df['N10_Grd_Prod_Pwr_InternalDerateStat']<4) & #this removes curtailed values
    (final_df['M10_Grd_Prod_Pwr_InternalDerateStat']<4) &
    (final_df['N10_Amb_WindDir_Abs_Avg']>=angle_lower) & #220 degrees as the turbines of interest are aligned along this plane for wind dir
    (final_df['N10_Amb_WindDir_Abs_Avg']< angle_higher)][[
        'N10_Grd_Prod_Pwr_Avg',
        'M10_Grd_Prod_Pwr_Avg',
        'N10_Amb_WindSpeed_Avg',
        'M10_Amb_WindSpeed_Avg',
        'N10_Amb_WindDir_Abs_Avg',
        'WindSpeed_Mean',
        ]].copy()

    upstream_turbine_power = lead+'_Grd_Prod_Pwr_Avg'
    upstream_turbine_windspeed = lead+'_Amb_WindSpeed_Avg'
    print('power_df length'+ str(len(power_df)))

    #N10 windspeed correction via powercurve 
    corrected_Upstream_windspeed = []
    for row in power_df.itertuples(index=False):
        if getattr(row, upstream_turbine_power) < 3450:
            index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-getattr(row, upstream_turbine_power)))
            correct_windspeed = powercurve_windspeed_new[index_speed]
            if correct_windspeed < 3:
                corrected_Upstream_windspeed.append(0)

        if getattr(row, upstream_turbine_power) < 3450:
            index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-getattr(row, upstream_turbine_power)))
            correct_windspeed = powercurve_windspeed_new[index_speed]
            corrected_Upstream_windspeed.append(correct_windspeed)
        if getattr(row, upstream_turbine_power) >= 3450:
            correct_windspeed = getattr(row, upstream_turbine_windspeed)
            corrected_Upstream_windspeed.append(correct_windspeed)

    print(len(corrected_Upstream_windspeed))
    power_df["corrected_Upstream_windspeed"] = corrected_Upstream_windspeed

    #M10 windspeed correction via powercurve 
    corrected_downstream_windspeed = []     
    downstream_turbine_power = behind+'_Grd_Prod_Pwr_Avg'
    downstream_turbine_windspeed = behind+'_Amb_WindSpeed_Avg'
    
    for row in power_df.itertuples():

        if getattr(row, downstream_turbine_power) < 3450:
            index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-getattr(row, downstream_turbine_power)))
            correct_windspeed = powercurve_windspeed_new[index_speed]
            if correct_windspeed < 3:
                corrected_downstream_windspeed.append(0)

        if getattr(row, downstream_turbine_power) < 3450:
            index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-getattr(row, downstream_turbine_power)))
            correct_windspeed = powercurve_windspeed_new[index_speed]
            corrected_downstream_windspeed.append(correct_windspeed)
        if getattr(row, downstream_turbine_power) >= 3450:
            correct_windspeed = getattr(row, downstream_turbine_windspeed)
            corrected_downstream_windspeed.append(correct_windspeed)

    power_df["corrected_downstream_windspeed"] = corrected_downstream_windspeed

    plt.figure(100)

    upstream_windspeed_corrected = np.asarray(power_df['corrected_Upstream_windspeed'])
    downstream_windspeed_corrected = np.asarray(power_df['corrected_downstream_windspeed'])

    upstream_windspeed_corrected = upstream_windspeed_corrected.reshape(-1,1)
    downstream_windspeed_corrected = downstream_windspeed_corrected.reshape(-1,1)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(upstream_windspeed_corrected, downstream_windspeed_corrected)



    line_X = np.arange(upstream_windspeed_corrected.min(), upstream_windspeed_corrected.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    plt.scatter(upstream_windspeed_corrected,downstream_windspeed_corrected,
    marker='x',s=5, label='Jensen Windspeed Prediction')
    plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=2,
            label='RANSAC regressor')

    plt.legend(loc="upper left")


    # print(line_X)

    plt.show()

def main(lead,behind):
    print('test')
    get_ransac(lead,behind)


if __name__ =='__main__':
    main(sys.argv[1],sys.argv[2])





# what to do:
# 1) assume all similar distances
# 2) List of turbines to do should be arond 30 for that probability law

# 226 degrees bearing
# J02 I02
# K03 J03
# L04 L04
# L05 K05
# N06 M06
# O09 N09
# N10 M10 
# K11 J11
# K12 J12
# K13 J13
# J14 I14
# I15 H15

# 106 degree bearing
# A10 A09
# B10 B09
# C12 C11
# D13 D12
# E14 E13
# F15 F14
# G15 G14
# H15 H14
# I15 I14
# J14 J13
# K13 K12
# L10 L09
# M10 M09
# N10 N09

# 46 degrees
# B07 C07
# A08 B08
# A09 B09
# A10 B10
# C11 D11
# C12 D12
# D13 E13
