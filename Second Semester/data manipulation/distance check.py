from math import sin, cos, sqrt, atan2, radians
from geopy import distance

from geopy.distance import geodesic

import numpy as np

import csv
import sys

# get the distances between A, B and C 
# # feed these distances into the Jensen Line Calculations
# and smooth it out for wind speeds 0<25 for the turbines
# # convert this jensen line into a power line



# jensen_x_axis = []
# jensen_y_axis = []
# x_new,smooth_jensen_speed
# for i in range(0,len(x_new)):
#     print(i)
#     power = return_power_value(x_new[i],powercurve_windspeed_new,power_smooth)
#     jensen_x_axis.append(power)

#     power = return_power_value(smooth_jensen_speed[i],powercurve_windspeed_new,power_smooth)
#     jensen_y_axis.append(power)


def return_power_value(speed,powercurve_windspeed_new,power_smooth):
    index_power = min(range(len(powercurve_windspeed_new)), key=lambda i: abs(powercurve_windspeed_new[i]-speed))
    power_value = power_smooth[index_power]
    return power_value

def get_distances(dir):


    turbine_list = csv.reader(open('turbine_list_'+dir+'_2wakes.txt', "r"), delimiter=",")
    turbine_gps = csv.reader(open('turbine_GPS.txt', "r"), delimiter=",")
    next(turbine_list)

    distance_list = []

    #loop through the csv list
    for row in turbine_list:
        lead_turbine = row[0]
        middle_turbine = row[1]
        rear_turbine = row[2]
        # print(row)
        
        turbine_gps = csv.reader(open('turbine_GPS.txt', "r"), delimiter=",")
        next(turbine_gps)
        for line in turbine_gps:
            if line[0] == lead_turbine:
                lead_turbine_lat = str(line[1])
                lead_turbine_lon = str(line[2])
            if line[0] == rear_turbine:
                rear_turbine_lat = str(line[1])
                rear_turbine_lon = str(line[2])
            if line[0] == middle_turbine:
                middle_turbine_lat = str(line[1])
                middle_turbine_lon = str(line[2])

        # print(rear_turbine_lat)
        # print(lead_turbine + ' ' + rear_turbine)
        # print(lead_turbine_lat + ' ' + lead_turbine_lon)
        # print(rear_turbine_lat + ' ' + rear_turbine_lon)
        lead_coords = (lead_turbine_lat,lead_turbine_lon)
        rear_coords = (rear_turbine_lat,rear_turbine_lon)
        middle_coords = (middle_turbine_lat,middle_turbine_lon)
        distance_middle_rear = geodesic(middle_coords, rear_coords).meters
        distance_lead_rear = geodesic(lead_coords, rear_coords).meters

        distance_list.append([distance_lead_rear,distance_middle_rear])

    return distance_list
distances = []
distances_46 = get_distances('46')
distances_106 = get_distances('106')
distances_226 = get_distances('226')





# do frandsen
#do two wakes
# do the two wake thing to compare the parameters, and check the distance thing you get me
    