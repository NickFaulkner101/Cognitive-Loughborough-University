import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import make_interp_spline, BSpline
from sklearn import linear_model, datasets
import sys
import csv
import os


from geopy import distance
from geopy.distance import geodesic




def return_power_value(speed,powercurve_windspeed_new,power_smooth):
    index_power = min(range(len(powercurve_windspeed_new)), key=lambda i: abs(powercurve_windspeed_new[i]-speed))
    power_value = power_smooth[index_power]
    return power_value



def get_smoothed_jensen_line(distance):


    x = np.linspace(0,25,26) #up to max wind speed

    jensen_speed = []
    single_wake_data = []

    for i in range(0,len(x)):
        rd=56
        kw = 0.04
        turbine_distance = float(distance)
        Ct = get_ct_value(i)
        factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
        jensen_speed.append(factor)
        single_wake_data.append(factor*i)


    x_smoothed = np.linspace(x.min(), x.max(), 3000)
    spl = make_interp_spline(x, single_wake_data, k=1)
    jensen_smoothed = spl(x_smoothed)



    #powercurve load in

    # plt.plot(x,single_wake_data)
    # plt.plot(x_smoothed,jensen_smoothed)
    # plt.grid()
    # plt.show()

    # smoothed_jensen_line = [x_new,smooth_jensen_speed]

  


    return [x_smoothed,jensen_smoothed] 



def get_jensen_speed(correct_windspeed,jensen_line):
    
    index_power = min(range(len(jensen_line[0])), key=lambda i: abs(jensen_line[0][i]-correct_windspeed))
    downstream_speed = jensen_line[1][index_power]
    # print(str(downstream_speed) + '  ' + str(correct_windspeed))
    return downstream_speed


def get_dataframe(lead,behind,third,angle_lower,angle_higher,speed):
    # Load in WindSpeed Data
    df = pd.read_csv("WindSpeed_Average.csv")
    df.index=df['timestamp']
    df = df.drop('timestamp', axis =1)
    df['WindSpeed_Mean'] = df.mean(axis=1)


    distance = get_distances(lead,behind)
    print(distance)

    jensen_line = get_smoothed_jensen_line(distance)

    

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
    powercurve_windspeed= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13])
    powercurve_windspeed_new = np.linspace(powercurve_windspeed.min(), powercurve_windspeed.max(), 500)  
    powercurve_power = [7,53,123,208,309,427,567,732,927,1149,1401,1688,2006,2348,2693,3011,3252,3388,3436,3448,3450]


    spl = make_interp_spline(powercurve_windspeed, powercurve_power, k=1)  # type: BSpline
    #powercurve load in
    power_smooth = spl(powercurve_windspeed_new)
    # plt.figure(3)
    # plt.plot(powercurve_windspeed_new, power_smooth,label='New')
    # plt.legend(loc="upper left")
    # plt.grid()
    # plt.show()
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
    

    # (final_df[lead+'_Grd_Prod_Pwr_Avg'] > 0) &   
    # (final_df[behind+'_Grd_Prod_Pwr_Avg'] > 0) & #this removes null values that pandas struggles with
    # (final_df[third+'_Grd_Prod_Pwr_Avg'] > 0) &
    # (final_df[lead+'_Grd_Prod_Pwr_InternalDerateStat']<4) & #this removes curtailed values
    # (final_df[behind+'_Grd_Prod_Pwr_InternalDerateStat']<4) &
    # (final_df[third+'_Grd_Prod_Pwr_InternalDerateStat']<4) &
    (final_df[lead+'_Amb_WindDir_Abs_Avg']>=angle_lower) & 
    (final_df[lead+'_Amb_WindDir_Abs_Avg']< angle_higher) &
    (final_df[lead+'_Amb_WindSpeed_Avg']>=speed-0.5) & 
    (final_df[lead+'_Amb_WindSpeed_Avg'] <speed+0.5)][[
        lead+'_Grd_Prod_Pwr_Avg',
        behind+'_Grd_Prod_Pwr_Avg',
        third+'_Grd_Prod_Pwr_Avg',
        lead+'_Amb_WindSpeed_Avg',
        behind+'_Amb_WindSpeed_Avg',
        third+'_Amb_WindSpeed_Avg',
        lead+'_Amb_WindDir_Abs_Avg',
        'WindSpeed_Mean',
        ]].copy()

    print('Sample size: '+ str(len(power_df)))

    power_df.to_csv(r'Dataframe_'+lead+'_'+behind+'DualWakeScenario.csv', header=True)
    return 

    upstream_turbine_power = lead+'_Grd_Prod_Pwr_Avg'
    upstream_turbine_windspeed = lead+'_Amb_WindSpeed_Avg'
    


    #N10 windspeed correction via powercurve 
    corrected_Upstream_windspeed = []
    jensen_predicted_windspeed = []
    jensen_predicted_power = []

    print('Correcting upstream wind speed measurements via power curve')
    for row in power_df.itertuples(index=False):
        if getattr(row, upstream_turbine_power) < 3450:
            index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-getattr(row, upstream_turbine_power)))
            correct_windspeed = powercurve_windspeed_new[index_speed]
            if correct_windspeed < 4:
                corrected_Upstream_windspeed.append(getattr(row, upstream_turbine_windspeed))
            if correct_windspeed > 4:
                index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-getattr(row, upstream_turbine_power)))
                correct_windspeed = powercurve_windspeed_new[index_speed]
                corrected_Upstream_windspeed.append(correct_windspeed)
        if getattr(row, upstream_turbine_power) >= 3450:
            correct_windspeed = getattr(row, upstream_turbine_windspeed)
            corrected_Upstream_windspeed.append(correct_windspeed)
            


        
        predicted_downstream = get_jensen_speed(correct_windspeed,jensen_line) #get downstream Turbine B windspeed from Turbine A windspeed
        jensen_power = return_power_value(predicted_downstream,powercurve_windspeed_new,power_smooth)
        jensen_predicted_windspeed.append(predicted_downstream)
        jensen_predicted_power.append(jensen_power)

    power_df["power_inferred_"+lead+"_windspeed"] = corrected_Upstream_windspeed
    power_df["jensen_"+behind+"_windspeed"] = jensen_predicted_windspeed
    power_df["jensen_"+behind+"_power"] = jensen_predicted_power

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
            if correct_windspeed < 4:
                corrected_downstream_windspeed.append(getattr(row, downstream_turbine_windspeed))
            if correct_windspeed > 4:
                index_speed = min(range(len(power_smooth)), key=lambda i: abs(power_smooth[i]-getattr(row, downstream_turbine_power)))
                correct_windspeed = powercurve_windspeed_new[index_speed]
                corrected_downstream_windspeed.append(correct_windspeed)
        if getattr(row, downstream_turbine_power) >= 3450:
            correct_windspeed = getattr(row, downstream_turbine_windspeed)
            corrected_downstream_windspeed.append(correct_windspeed)



    print(power_df.shape[0])
    print(len(corrected_downstream_windspeed))

    power_df["power_inferred_"+behind+"_windspeed"] = corrected_downstream_windspeed

    print('Saving Dataframe...')

    # now we want to append to each row, the predicted downsream speed, and the predicted downstream power




    return


def get_avg(lead,mid,third,angle_lower,angle_higher,speed):
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

    #Merge Dataframes
    new_df=df.merge(df_Dir,left_index=True,right_index=True)
    new_new_df=new_df.merge(df_curtailed,left_index=True,right_index=True)
    final_df=new_new_df.merge(df_power,left_index=True,right_index=True)

    angle_lower = angle_lower
    angle_higher = angle_higher

    power_df=final_df.loc[
    

    (final_df[lead+'_Grd_Prod_Pwr_Avg'] > 0) &   
    (final_df[mid+'_Grd_Prod_Pwr_Avg'] > 0) & #this removes null values that pandas struggles with
    (final_df[third+'_Grd_Prod_Pwr_Avg'] > 0) &
    (final_df[lead+'_Grd_Prod_Pwr_InternalDerateStat']<4) & #this removes curtailed values
    (final_df[mid+'_Grd_Prod_Pwr_InternalDerateStat']<4) &
    (final_df[third+'_Grd_Prod_Pwr_InternalDerateStat']<4) &
    (final_df[lead+'_Amb_WindDir_Abs_Avg']>=angle_lower) & 
    (final_df[lead+'_Amb_WindDir_Abs_Avg']< angle_higher) &
    (final_df[lead+'_Amb_WindSpeed_Avg']>=speed-0.5) & 
    (final_df[lead+'_Amb_WindSpeed_Avg'] <speed+0.5)][[
        lead+'_Grd_Prod_Pwr_Avg',
        mid+'_Grd_Prod_Pwr_Avg',
        third+'_Grd_Prod_Pwr_Avg',
        ]].copy()


    return [np.asarray(final_df[lead+'_Grd_Prod_Pwr_Avg']),np.asarray(final_df[mid+'_Grd_Prod_Pwr_Avg']).np.asarray(final_df[third+'_Grd_Prod_Pwr_Avg'])]



def get_ct_value(speed):

    #modified values
    windspeed_values= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
    windspeed_values_new = np.linspace(windspeed_values.min(), windspeed_values.max(), 100)  
    # ctvalues = [0.691,0.685,0.665,0.639,0.640,0.640,0.634,0.627,0.620,0.612,0.603,0.601,0.696,0.684,0.53,0.698,0.627,0.546,0.469,0.403,0.391,0.350,0.3,0.355,0.219,0.298,0.280,0.264,0.250,0.238,0.227,0.217,0.208,0.200,0.193,0.187,0.181,0.177,0.172,0.168,0.164,0.160,0.157,0.154,0.151]

    # ctvalues = 0.6*np.ones(len(windspeed_values)) #ct values that are constant

    #original actual turbine values
    # windspeed_values= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
    # windspeed_values_new = np.linspace(windspeed_values.min(), windspeed_values.max(), 100)  
    ctvalues = [0.891,0.885,0.865,0.839,0.840,0.840,0.834,0.827,0.820,0.812,0.803,0.801,0.796,0.784,0.753,0.698,0.627,0.546,0.469,0.403,0.351,0.310,0.275,0.245,0.219,0.198,0.180,0.164,0.150,0.138,0.127,0.117,0.108,0.100,0.093,0.087,0.081,0.077,0.072,0.068,0.064,0.060,0.057,0.054,0.051]


    spl = make_interp_spline(windspeed_values, ctvalues, k=1)  # type: BSpline
    ct_values_smoothed = spl(windspeed_values_new)

    index_speed = min(range(len(windspeed_values_new)), key=lambda i: abs(windspeed_values_new[i]-speed))
    returned_ct = ct_values_smoothed[index_speed]

    return returned_ct



def get_distances(lead_turbine,rear_turbine):



    turbine_gps = csv.reader(open('turbine_GPS.txt', "r"), delimiter=",")
    next(turbine_gps)

    for line in turbine_gps:
        if line[0] == lead_turbine:
            lead_turbine_lat = str(line[1])
            lead_turbine_lon = str(line[2])
        if line[0] == rear_turbine:
            rear_turbine_lat = str(line[1])
            rear_turbine_lon = str(line[2])
        

    lead_coords = (lead_turbine_lat,lead_turbine_lon)
    rear_coords = (rear_turbine_lat,rear_turbine_lon)
    distance_lead_rear = geodesic(lead_coords, rear_coords).meters

    

    return distance_lead_rear


def getavg():
    lead_power_array = []
    mid_power_array = []
    third_power_array = []
    turbine_list = csv.reader(open('turbine_list_46_2wakes.txt', "r"), delimiter=",")
    next(turbine_list)

    for row in turbine_list:
        lead = row[0]
        mid = row[1]
        third = row[2]
        returned_power_list = get_avg(lead,mid,third,44,48,8)
        print(returned_power_list)
        lead_power_array.extend(returned_power_list[0])
        mid_power_array.extend(returned_power_list[1])
        third_power_array.extend(returned_power_list[2])

    print(third_power_array)
    print(np.average(third_power_array))

    return

    turbine_list = csv.reader(open('turbine_list_106_2wakes.txt', "r"), delimiter=",")
    next(turbine_list)

    for row in turbine_list:
        lead = row[0]
        mid = row[1]
        third = row[2]
        returned_power_list = get_avg(lead,mid,third,44,48,8)
        print(returned_power_list)
        lead_power_array.extend(returned_power_list[0])
        mid_power_array.extend(returned_power_list[1])
        third_power_array.extend(returned_power_list[2])


    turbine_list = csv.reader(open('turbine_list_46_2wakes.txt', "r"), delimiter=",")
    next(turbine_list)

    for row in turbine_list:
        lead = row[0]
        mid = row[1]
        third = row[2]
        returned_power_list = get_avg(lead,mid,third,44,48,8)
        print(returned_power_list)
        lead_power_array.append(returned_power_list[0])
        mid_power_array.append(returned_power_list[1])
        third_power_array.append(returned_power_list[2])


    third_avg = np.average(third_power_array)


def main():

    getavg()
    return


    speed = 8
    turbine_list = csv.reader(open('turbine_list_46_2wakes.txt', "r"), delimiter=",")
    next(turbine_list)

    for row in turbine_list:
        lead_turbine = row[0]
        rear_turbine = row[1]
        third = row[2]
        get_dataframe(str(lead_turbine),str(rear_turbine),str(third),43,49,speed)

    turbine_list = csv.reader(open('turbine_list_226_2wakes.txt', "r"), delimiter=",")
    next(turbine_list)

    for row in turbine_list:
        lead_turbine = row[0]
        rear_turbine = row[1]
        third = row[2]
        get_dataframe(str(lead_turbine),str(rear_turbine),223,229)

    turbine_list = csv.reader(open('turbine_list_106_2wakes.txt', "r"), delimiter=",")
    next(turbine_list)

    for row in turbine_list:
        lead_turbine = row[0]
        rear_turbine = row[1]
        third = row[2]
        get_dataframe(str(lead_turbine),str(rear_turbine),103,109)
    


    return




if __name__ =='__main__':
    main()
