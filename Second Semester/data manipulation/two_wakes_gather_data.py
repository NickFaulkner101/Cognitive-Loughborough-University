# 1) Understand RANSAC, do a ransac
# 2) compare ransac to the 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import make_interp_spline, BSpline
from sklearn import linear_model, datasets
import sys
import csv
import os


#all we need to do is gather data for lead and last, and save this



def get_ransac(lead,behind,angle_lower,angle_higher):
    # Load in WindSpeed Data
    df = pd.read_csv("WindSpeed_Average.csv")
    df.index=df['timestamp']
    df = df.drop('timestamp', axis =1)
    df['WindSpeed_Mean'] = df.mean(axis=1)

    print('Upstream: ' + lead + ' Downstream: ' + behind + ' Wind Angle Between: ' + str(angle_lower) + ' & ' + str(angle_higher))

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
    angle_lower = angle_lower
    angle_higher = angle_higher

    power_df=final_df.loc[
    

    (final_df[lead+'_Grd_Prod_Pwr_Avg'] > 0) &   
    (final_df[behind+'_Grd_Prod_Pwr_Avg'] > 0) & #this removes null values that pandas struggles with
    
    (final_df[lead+'_Grd_Prod_Pwr_InternalDerateStat']<4) & #this removes curtailed values
    (final_df[behind+'_Grd_Prod_Pwr_InternalDerateStat']<4) &
    (final_df[lead+'_Amb_WindDir_Abs_Avg']>=angle_lower) & #220 degrees as the turbines of interest are aligned along this plane for wind dir
    (final_df[lead+'_Amb_WindDir_Abs_Avg']< angle_higher)][[
        lead+'_Grd_Prod_Pwr_Avg',
        behind+'_Grd_Prod_Pwr_Avg',
        lead+'_Amb_WindSpeed_Avg',
        behind+'_Amb_WindSpeed_Avg',
        lead+'_Amb_WindDir_Abs_Avg',
        'WindSpeed_Mean',
        ]].copy()

    print('Sample size: '+ str(len(power_df)))

    upstream_turbine_power = lead+'_Grd_Prod_Pwr_Avg'
    upstream_turbine_windspeed = lead+'_Amb_WindSpeed_Avg'
    

    #N10 windspeed correction via powercurve 
    corrected_Upstream_windspeed = []
    print('Correcting upstream wind speed measurements via power curve')
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

    power_df["corrected_Upstream_windspeed"] = corrected_Upstream_windspeed

    #M10 windspeed correction via powercurve 
    corrected_downstream_windspeed = []     
    downstream_turbine_power = behind+'_Grd_Prod_Pwr_Avg'
    downstream_turbine_windspeed = behind+'_Amb_WindSpeed_Avg'
    
    print('Correcting downstream wind speed measurements via power curve')
    print('')
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

    # plt.figure(100)
    print('Calculating regression...')

    upstream_windspeed_corrected = np.asarray(power_df['corrected_Upstream_windspeed'])
    downstream_windspeed_corrected = np.asarray(power_df['corrected_downstream_windspeed'])

    upstream_windspeed_corrected = upstream_windspeed_corrected.reshape(-1,1)
    downstream_windspeed_corrected = downstream_windspeed_corrected.reshape(-1,1)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(upstream_windspeed_corrected, downstream_windspeed_corrected)

    print('RANSAC estimator coefficient: ' + str(ransac.estimator_.coef_))
    print('')
    print('')
    print('')

# just want to append the raw x and y values (the upstream values etc)

    line_X = np.arange(upstream_windspeed_corrected.min(), upstream_windspeed_corrected.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    # plt.scatter(upstream_windspeed_corrected,downstream_windspeed_corrected,
    # marker='x',s=5, label='Jensen Windspeed Prediction')
    # plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=2,
    #         label='RANSAC regressor')

    # plt.legend(loc="upper left")

    return [line_X,line_y_ransac,upstream_windspeed_corrected,downstream_windspeed_corrected,np.asarray(power_df[upstream_turbine_power]),np.asarray(power_df[downstream_turbine_power])]

    # print(line_X)

    # plt.show()

def save(path,file,name):
    
    name = str(name)

    np.savez(name,array=file)
    
    # os.makedirs(str(path),exist_ok=True)
    # np.savetxt(str(path)+"/"+str(name)+".csv", file, delimiter=",",fmt='%s')

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



def main():
    print('Starting Data Collection ...')
    print('')

    upstream_array = []
    downstream_array = []
    
    turbine_list_226 = []
    # Load in 226 Degrees
    with open('turbine_list_226_2wakes.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_file)
        for row in csv_reader:
            turbine_list_226.append([row[0],row[2]])
    
    #get RANSACs for 226 degrees
    ransac_lists = []
    saved_list_226 = []
    windspeed_226_upstream = []
    windspeed_226_downstream = []
    power_226_upstream = []
    power_226_downstream = []
    for i in range(0,len(turbine_list_226)):
        angle_lower = 224
        angle_higher = 228
        returned_ransac = get_ransac(turbine_list_226[i][0],turbine_list_226[i][1],angle_lower,angle_higher)
        ransac_lists.append(returned_ransac)
        saved_list_226.append(returned_ransac)

        upstream_array.extend(returned_ransac[2])

        windspeed_226_upstream.append(returned_ransac[2])
        windspeed_226_downstream.append(returned_ransac[3])

        downstream_array.extend(returned_ransac[3])

        power_226_upstream.append(returned_ransac[4])
        power_226_downstream.append(returned_ransac[5])

    



        # save(path,file,'226_degree'):
        # make a new list for each angle direction, append to each

    turbine_list_106 = []
    # Load in 106 Degrees
    with open('turbine_list_106_2wakes.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_file)
        for row in csv_reader:
            turbine_list_106.append([row[0],row[2]])
    
    #get RANSACs for 106 degrees
    ransac_lists_106 = []
    saved_list_106 = []
    windspeed_106_upstream = []
    windspeed_106_downstream = []
    power_106_upstream = []
    power_106_downstream = []
    for i in range(0,len(turbine_list_106)):
        angle_lower = 104
        angle_higher = 106
        returned_ransac = get_ransac(turbine_list_106[i][0],turbine_list_106[i][1],angle_lower,angle_higher)
        ransac_lists_106.append(returned_ransac)
        upstream_array.extend(returned_ransac[2])
        saved_list_106.append(returned_ransac)
        downstream_array.extend(returned_ransac[3])

        windspeed_106_upstream.append(returned_ransac[2])
        windspeed_106_downstream.append(returned_ransac[3])

        power_106_upstream.append(returned_ransac[4])
        power_106_downstream.append(returned_ransac[5])

    turbine_list_46 = []
    # Load in 46 Degrees
    with open('turbine_list_46_2wakes.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_file)
        for row in csv_reader:
            print('row ' + str(row))
            turbine_list_46.append([row[0],row[2]])
    
    #get RANSACs for 46 degrees
    ransac_lists_46 = []
    saved_list_46 = []
    windspeed_46_upstream = []
    windspeed_46_downstream = []
    power_46_upstream = []
    power_46_downstream = []
    for i in range(0,len(turbine_list_46)):
        angle_lower = 44
        angle_higher = 46
        returned_ransac = get_ransac(turbine_list_46[i][0],turbine_list_46[i][1],angle_lower,angle_higher)
        ransac_lists_46.append(returned_ransac)
        saved_list_46.append(returned_ransac)
        upstream_array.extend(returned_ransac[2])
        downstream_array.extend(returned_ransac[3])

        windspeed_46_upstream.append(returned_ransac[2])
        windspeed_46_downstream.append(returned_ransac[3])

        power_46_upstream.append(returned_ransac[4])
        power_46_downstream.append(returned_ransac[5])

    #save data for rapid graphing later

    path = './saved_data'


    plt.figure(600)

    plt.scatter(power_226_upstream,power_226_downstream)
    plt.show()

    save(path,saved_list_226,'226_degree_2wakes')
    save(path,saved_list_106,'106_degree_2wakes')
    save(path,saved_list_46,'46_degree_2wakes')

    # save(path,power_226_upstream,'226_power_up_2wakes')
    # save(path,power_226_downstream,'226_power_down_2wakes')
    # save(path,power_46_upstream,'46_power_up_2wakes')
    # save(path,power_46_downstream,'46_power_down_2wakes')
    # save(path,power_106_upstream,'106_power_up_2wakes')
    # save(path,power_106_downstream,'106_power_down_2wakes')
   

        # make a new list for each angle direction, append to each


    
    # #plot 226 degree turbines
    # for i in range(0,len(ransac_lists)):
    #     plt.plot(ransac_lists[i][0], ransac_lists[i][1], color='cornflowerblue', linewidth=2,
    #         label=turbine_list_226[i][0]+' & '+turbine_list_226[i][1])
    
    # #plot 106 degree turbines
    # for i in range(0,len(ransac_lists_106)):
    #     plt.plot(ransac_lists_106[i][0], ransac_lists_106[i][1], color='darkmagenta', linewidth=2,
    #         label=turbine_list_106[i][0]+' & '+turbine_list_106[i][1])
    
    # #plot 46 degree turbines
    # for i in range(0,len(ransac_lists_46)):
    #     plt.plot(ransac_lists_46[i][0], ransac_lists_46[i][1], color='orangered', linewidth=2,
    #         label=turbine_list_46[i][0]+' & '+turbine_list_46[i][1])
    
    # # plt.legend(loc="upper left")
    # plt.xlabel("Upstream Wind Turbine Corrected Wind Speed (m/s)")
    # plt.ylabel("Downstream Wind Turbine Corrected Wind Speed (m/s)")
    # plt.title("RANSAC for 30 Turbine Wake Single-Wake Relationships")




    upstream_array = np.array(upstream_array)
    downstream_array = np.array(downstream_array)

    # plt.figure(100)
    # plt.scatter(upstream_array,downstream_array,s=1)


    # plt.figure(101)
    # plt.scatter(windspeed_226_upstream,windspeed_226_downstream,color='cornflowerblue',marker='x',s=1,label='226 Degrees Data')
    # plt.scatter(windspeed_106_upstream,windspeed_106_downstream,marker='x',s=1,color='darkmagenta',label='106 Degrees Data')
    # plt.scatter(windspeed_46_upstream,windspeed_46_downstream, marker='x',s=1,color='orangered',label ='46 Degrees Data')
    # plt.legend(loc="upper left")
    # plt.grid()
    # plt.title('Lead turbine wind speed versus downstream turbine wind speed behind 2 turbines')
    # plt.xlabel('Upstream Wind Speed (m/s)')
    # plt.ylabel('Downstream Wind Speed (m/s)')






    x = np.linspace(0,25,26)

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


    x_new = np.linspace(x.min(), x.max(), 100)

    spl = make_interp_spline(x, overall_jensen_speed, k=1)  # type: BSpline
    #powercurve load in
    smooth_jensen_speed = spl(x_new)

    

    powercurve_windspeed= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5])
    powercurve_windspeed_new = np.linspace(powercurve_windspeed.min(), powercurve_windspeed.max(), 3000)  
    powercurve_power = [7,53,123,208,309,427,567,732,927,1149,1401,1688,2006,2348,2693,3011,3252,3388,3436,3448,3450]

    spl = make_interp_spline(powercurve_windspeed, powercurve_power, k=1)  # type: BSpline
    #powercurve load in
    power_smooth = spl(powercurve_windspeed_new)


    jensen_x_axis = []
    jensen_y_axis = []
    x_new,smooth_jensen_speed
    for i in range(0,len(x_new)):
        print(i)
        power = return_power_value(x_new[i],powercurve_windspeed_new,power_smooth)
        jensen_x_axis.append(power)

        power = return_power_value(smooth_jensen_speed[i],powercurve_windspeed_new,power_smooth)
        jensen_y_axis.append(power)



   
    save(path,jensen_x_axis,'jensen_power_x_2wakes')
    save(path,jensen_y_axis,'jensen_power_y_2wakes')


    save(path,upstream_array,'upstream_array_2wakes')
    save(path,downstream_array,'downstream_array_2wakes')


if __name__ =='__main__':
    main()






# qq plot? compare validity/skewness of Jensen to assess it