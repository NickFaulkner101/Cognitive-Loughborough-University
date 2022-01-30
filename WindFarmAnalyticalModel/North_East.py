import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import time
import csv
import sys
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import PolyCollection
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.path import Path
from matplotlib import cm
from random import randrange

from scipy.interpolate import make_interp_spline, BSpline

import os




#vestas v112 3.5MW power curve
#  https://www.vestas.com/en/products/4-mw-platform/v112-3_45_mw#!technical-specifications

#  https://www.sciencedirect.com/science/article/pii/S0960148116309429?via%3Dihub



#----------------------Each Turbine Wake is bound by 2 lines expanding down-wind----------------------
def equation_of_line1(x,turbine_coordinates, radius, direction):
    turbine_x = turbine_coordinates[0]
    turbine_y = turbine_coordinates[1]
    
    k_w = -0.04 
    line2_point2_x =  x*np.cos(-direction) - (-radius+k_w*x)*np.sin(-direction) + turbine_x
    line2_point2_y =  x*np.sin(-direction) + (-radius+k_w*x)*np.cos(-direction) + turbine_y
    
    return [line2_point2_x, line2_point2_y]
    
def equation_of_line2(x, turbine_coordinates, radius, direction):
    turbine_x = turbine_coordinates[0]
    turbine_y = turbine_coordinates[1]
    
    k_w = +0.04 
    line1_point2_x =  x*np.cos(-direction) - (radius+k_w*x)*np.sin(-direction) + turbine_x
    line1_point2_y =  x*np.sin(-direction) + (radius+k_w*x)*np.cos(-direction) + turbine_y

    return [line1_point2_x, line1_point2_y]

#------------------Starting Point of Each Wake, Geometry Calculated here depending on wind angle---------------------------
def wake_start_points(coordinates,angle,r):
    theta = (angle) * np.pi / 180.
    #               -----------------------------------------------------------------------------------------------------------------
                    # wake outline starts from the diameter of the turbine blades                                           (wake)
                    # as the nacelle rotates to face the wind, the coordinates of the wake starting points change  i.e.   \        /
                                                                                                                        #  \      /
                                                                                                                        #   \    /
                                                                                                                        #    \  / 
    #                                              s                                                                         ^  ^ these points rotate
    #               Let the points (x1,y1) and (x2,y2) represent the two points on edge of turbine swept area
                    # the 4 variables are affectede by cos(theta) and sin(theta) depending on whether theta is 0<t<90, 90<t<180, -90<t<0, -180<t<-90
                    # where x and y are the turbine coodrinates
                    # i.e. for 0<theta<90 (up to coming from east)  x1 = x - rcos(theta)    y1 = y + rsin(theta)   x2 = x + rcos(theta) y2 = y -rsin(theta)
    #               ------------------------------------------------------------------------------------------------------------------

    x = float(coordinates[0])
    y = float(coordinates[1])
    # print(type(angle))
    #sign coefficients to ensure wakes start in right position, considering cos and sin waves 
    
    A = -1
    B = 1 
    C = 1
    D = -1
    
    x1 = x + A*r*np.cos(theta)
    x2 = x + B*r*np.cos(theta)
    y1 = y + C*r*np.sin(theta)
    y2 = y + D*r*np.sin(theta)
    return [x1,x2,y1,y2]

def windspeed_probe(coordinates,angle,X,Y,r,Z):
#-----------------------------------------------------------------------------------------------------------
#       Wind speed for each respective turbine needs to be measured from a slight upstream position
#       to the wind turbine coordinates. If incident wind speed is measured from the wind turbine center itself,
#       the down stream speed in the wake is recorded.

#       Additionally, the probe is the average of two positions at the rotor edges, in case wake only covers part of turbine
#-----------------------------------------------------------------------------------------------------------     
    
    theta = (angle) * np.pi / 180.

    x = float(coordinates[0])
    y = float(coordinates[1])
    # print(type(angle))
    #sign coefficients to ensure wakes start in right position, considering cos and sin  

    # x_coord=  x+10*np.sin(theta)
    # y_coord = y+10*np.cos(theta)
    

    #turbine centre
    #pick 6 points along the turbine and calculate wind speed at each point
    #each point needs to be slightly upstream
    #then average the speeds
    #the below loop functions by working out the speed at 6 points radiating from the centre of the turbine
    speeds = []
    for i in range(1,5):
        #point 1 and point 2 are same distance either direction from wind turbine nacelle/centre
        x1 = x -(i/6)*r*np.cos(theta)
        x2 = x + (i/6)*r*np.cos(theta)
        y1 = y + (i/6)*r*np.sin(theta)
        y2 = y - (i/6)*r*np.sin(theta)

        x1_coord=  x1+10*np.sin(theta)
        y1_coord = y1+10*np.cos(theta)

        x2_coord=  x2+10*np.sin(theta)
        y2_coord = y2+10*np.cos(theta)


        x1_index=min(range(len(X[0])), key=lambda i: abs(X[0][i]-x1_coord))   # returmns closest x and y 
        y1_index =min(range(len(X[0])), key=lambda i: abs(Y[i][0]-y1_coord))

        x2_index=min(range(len(X[0])), key=lambda i: abs(X[0][i]-x2_coord))   # returmns closest x and y 
        y2_index =min(range(len(X[0])), key=lambda i: abs(Y[i][0]-y2_coord))

        wind_speed1 = Z[y1_index][x1_index] #how meshgrids are indexed
        wind_speed2 = Z[y2_index][x2_index]
        speeds.append(wind_speed1)
        speeds.append(wind_speed2)
    # print(speeds)
    return np.mean(speeds)

def f(x_object, y_object, poly, origin):
    turbine_x = origin[0]
    turbine_y = origin[1]
    Ct = 0.8
    rd=23.5
    kw = 0.04 
    print('Calculating Turbine Array of Speed Deficiency Coefficients \n for turbine location: ' + str(origin))
    result=np.empty((x_object.shape[0],y_object.shape[0]),dtype="float32")
    start = time.process_time()
    for i in range(0, x_object.shape[0]):
        for j in range(0, y_object.shape[0]):

            x = x_object[j][i]
            y = y_object[j][i]

            p1 = Point(x,y)
            if p1.within(poly) == True:
                distance = np.sqrt((x-turbine_x)**2+(y-turbine_y)**2) # should be linear distance from centre point?
                factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*distance/rd))**2))
                result[j,i] = factor
                # print(factor)
            
            if p1.within(poly) != True:
                result[j,i] = 1
    
    print('Time for solution: '+str(time.process_time() - start)+' seconds\n')
    return result

def get_ct_value_enhanced(speed):

    #modified values
    windspeed_values= np.array([2,2.5,3,3.25,3.375,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
    windspeed_values_new = np.linspace(windspeed_values.min(), windspeed_values.max(), 100)  
    #modified to fit the wind data
    ctvalues = [0,0,0,0,0,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.60,0.60,0.60,0.57,0.57,0.5,0.45,0.45,0.4,0.39,0.35,0.325,0.325,0.3,0.3,0.25,0.25,0.25,0.25,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    index_speed = min(range(len(windspeed_values)), key=lambda i: abs(windspeed_values[i]-speed))
    returned_ct = ctvalues[index_speed]
    return returned_ct

#----------- f2 returns the array of coefficients for a turbine------------------
#-----------x and y objects are just meshgrids for the coordinates------------------
def f2(X, Y, coords,origin,x1,y1,x2,y2,speed):
    # turbine_x = float(origin[0])
    # turbine_y = float(origin[1])

    print('Calculating Turbine Array of Speed Deficiency Coefficients \n for turbine location: ' + str(origin))
    start = time.process_time()

    Ct = get_ct_value_enhanced(speed)
    rd=23.5
    kw = 0.04 
    # V0 = float(25)
    points = np.c_[X.ravel(), Y.ravel()]
    mask = Path(coords).contains_points(points).reshape(X.shape) #true/false array of points in wake, shape of meshgrid
    result=np.ones((X.shape[0],Y.shape[0]),dtype="float32")

    p1 = np.asarray((x1,y1))
    p2=np.asarray((x2,y2))

    for i in range (X.shape[0]):
        for j in range (Y.shape[0]):
            point = mask[j][i]
            if point == True:
                x = X[j][i]
                y = Y[j][i]
                p3 = np.asarray((x,y))
                distance = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1) #crossproduct to find downstream distance from point to turbine
                factor = ((1-math.sqrt(1-Ct))/(1+(kw*distance/rd))**2)   #jensens model
                result[j,i] = factor
    print('Time for solution: '+str(time.process_time() - start)+' seconds\n')
    return result


def get_array_of_jensens_factor(wake_distance,turbine_origin,U_direction,r_0, X, Y,speed):
    
    wind_direction = (U_direction+90) * np.pi / 180. # plus 90 to align with 0 degrees as north

    # -----------1 refers to point 1, 2 refers to point 2, the two points represent either side of the turbine swept area
    x1, x2, y1, y2 = wake_start_points(turbine_origin,U_direction,r_0)
    line1_point2 = equation_of_line1(wake_distance,[x1,y1], r_0, wind_direction) #get the distant point of the wake

    line2_point2 = []
    line2_point2 = equation_of_line2(wake_distance,[x2,y2], r_0, wind_direction) #get the distant point of the wake

    # -------------Create a Polygon--------------
    coords = [(x1, y1), (line1_point2[0], line1_point2[1]), (line2_point2[0], line2_point2[1]), (x2, y2)]
    # plt.plot(*poly.exterior.xy)
    #-------------Check If Point is in polygon--------------
    # print(p1.within(poly))
    # plt.plot(line1_xValues, line1_yValues, 'b-')
    # plt.plot(line2_xValues, line2_yValues, 'r-')
    # plt.axis('scaled')


    
    Z = f2(X, Y, coords,turbine_origin,x1,y1,x2,y2,speed)
    # Z = f(X,Y, poly, turbine_origin)

    return Z

def power_curve(speed):
    power = 0.0676*speed**6 - 3.2433*speed**5 + 60.607*speed**4 - 565.82*speed**3 + 2830.8*speed**2 - 7083.5*speed + 6896.3
    return power




def return_power_value(speed):

    powercurve_windspeed= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5])
    powercurve_windspeed_new = np.linspace(powercurve_windspeed.min(), powercurve_windspeed.max(), 5000)  
    powercurve_power = [7,53,123,208,309,427,567,732,927,1149,1401,1688,2006,2348,2693,3011,3252,3388,3436,3450]

    spl = make_interp_spline(powercurve_windspeed, powercurve_power, k=3)  # type: BSpline
    #powercurve load in
    power_smooth = spl(powercurve_windspeed_new)
    index_power = min(range(len(powercurve_windspeed_new)), key=lambda i: abs(powercurve_windspeed_new[i]-speed))
    power_value = power_smooth[index_power]
    return power_value



def get_power(Z,coordinates,U_direction,X,Y,r):
    power_generation_kw = []
    print('Calculating generated power at each turbine')
    for i in range(0,len(coordinates)):
#-------------------------------------------------------------
#                   Estimated Wind Speed on Probe
        
        wind_speed = windspeed_probe(coordinates[i],U_direction,X,Y,r,Z)
        
#---------------------------------------------------------------
        if wind_speed < 3 or wind_speed > 25: #cut in and cut out
            power = 0
        elif wind_speed >= 3 and wind_speed <= 12.5: #power curve
            power = return_power_value(wind_speed)
        elif wind_speed > 12.5 and wind_speed <= 25: # max power
            power = 3450
        elif wind_speed > 25: # max power
            power = 0
        power_generation_kw.append(power)
    return power_generation_kw



def AEP(power_kw,speed):
    #wind turbine should produce this amount in the wind
    num_turbines=len(power_kw)
    power_freestream = return_power_value(speed)
    total_free_power_kw = num_turbines*power_freestream

    total_farm_generated = sum(power_kw) #total power of farm at the windspeed, in kWh

    loss = total_free_power_kw-total_farm_generated
    percentage_loss = (loss*100)/total_free_power_kw


    return [total_free_power_kw,total_farm_generated,percentage_loss]

def wake_areas(coordinates,X,Y,wake_distance,U_direction,r_0):
    wind_direction = (U_direction+90) * np.pi / 180
    result=np.zeros((X.shape[0],Y.shape[0]),dtype="float32")
    for i in range(0,len(coordinates)):
        print('Finding Mask for Turbine'+ str(i)+' out of '+str(len(coordinates)-1))
        
        # -------------Create a Polygon--------------
        x1, x2, y1, y2 = wake_start_points(coordinates[i],U_direction,r_0)
        line1_point2 = equation_of_line1(wake_distance,[x1,y1], r_0, wind_direction) #get the distant point of the wake

        line2_point2 = []
        line2_point2 = equation_of_line2(wake_distance,[x2,y2], r_0, wind_direction) #get the distant point of the wake
        coords = [(x1, y1), (line1_point2[0], line1_point2[1]), (line2_point2[0], line2_point2[1]), (x2, y2)]

        points = np.c_[X.ravel(), Y.ravel()]
        mask = Path(coords).contains_points(points).reshape(X.shape) #true/false array of points in wake, shape of meshgrid
        for i in range (X.shape[0]):
            for j in range (Y.shape[0]):
                if mask[j][i] == True:
                    result[j][i] = True
    return result          

def wake_loss(jensens_factors,areas_in_wake,X,Y): 
    jensens_factors = np.asarray(jensens_factors)
    result=np.ones((X.shape[0],Y.shape[0]),dtype="float32")
#                           A Simple Model for Cluster Efficiency - Jensen 1978
#--------this iterates through areas in wake and seeks the grid cells which are in at least 1 turbine wake
#--------for these cells in the wakes, it works out the energy deficit(1-V/U)^2 for multuple wakes, as per Jensen 1987---------------------
    print('Calculating final wake losses for each cell...')

    for i in range(0,X.shape[0]):
            print(str(i)+' out of ' + str(X.shape[0]))
            for j in range(0,Y.shape[0]):
                if areas_in_wake[j][i] == True: #if the grid cell is in the wake of any turbines, check it out
                    values = []
                    for w in range(0,len(jensens_factors)):
                        values.append(jensens_factors[w][j][i]) # these values are from each flow field calculated for each turbine,
                                                                    # for this grid cell coordinate
                    values = list(filter(lambda a: a != 1, values))  # get all the values than are not 1, i.e. have a wind speed deficiency  
                    values.sort(reverse=False)               #cells = 1 should not be used in the below equation
                    one_Z = values[0]                       #filtering out 1 is very important, as we are only interested in deficiency
                    for jensens in range(1,len(values)): #work from largest to smallest, there is an order to working out (1-V/U)^2 = (1-V1/U)^2 + (1-V2/U)^2
                        one_Z = np.sqrt(np.square(one_Z)+np.square(values[jensens])) # this is Root Sum of Squares
                        
                    result[j][i] = 1-one_Z
    print('Final Cell MeshGrid Completed')
    return result
        
def save(path,file,name):
        
    
    os.makedirs(path,exist_ok=True)
    np.savetxt(path+"/"+name+".csv", file, delimiter=",")

def main(angle,speed):

#------------------- "0 degrees is coming from due North."---------------------------
#------- "+90 degrees means the wind is coming from due East, -90 from due West"-----
    U_direction = float(angle)
    V0 = speed # wind speed in m/s

    print('----------Starting Simulation--------')

#--------Turbine Rotor Radius-------------
    r_0 = 56 

#-----load coordinates of turbines----
    coordinates = []
    turbine_name = []
    # with open('turbines.txt') as csv_file:
    with open('Turbine_List.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_file)
        for row in csv_reader:
            coordinates.append([row[1],row[2]])
            turbine_name.append(row[0])


#------------flow domain settings-----------
    wake_distance = 4000 #wake distance in metres

#-----------discretisation grid spacing
    cells = 1000
    x = np.linspace(0,12300, cells, endpoint = True) # x intervals
    y = np.linspace(0,8500,cells, endpoint = True) # y intervals
    # x = np.linspace(-1500,1500, cells, endpoint = True) # x intervals
    # y = np.linspace(-1500,1500,cells, endpoint = True) # y intervals
    X, Y = np.meshgrid(x,y)
    print('Successfully Generated Grid')

#-------Creates MeshGrid for Each Wind Turbine, where Z corresponds to the wind speed
    list_of_jensens_factors = []
    for i in range(0,len(coordinates)):
        print('Turbine '+str(i)+' out of '+str(len(coordinates)-1))
        coefficient_matrix = np.array(get_array_of_jensens_factor(wake_distance,coordinates[i],U_direction,r_0, X, Y,speed))
        list_of_jensens_factors.append(coefficient_matrix)


#-------(REDACTED) Used for Multiplication of Jensen Factors, now difference of squares is used etc
    # jensens_factors = np.ones((X.shape[0],Y.shape[0]),dtype="float32")
    # for i in range(0,len(list_of_jensens_factors)):
    #     jensens_factors = np.multiply(jensens_factors,list_of_jensens_factors[i])

    
    #--------------This returns a meshgrid, each element refers to whether it is in the wake of any turbine.
#--------------this grid is used to calculate wind speeds for multiple turbines. Either a 1 or 0
#-------------This saves time, preventing the need for each cell to be calculated over and over
    areas_in_wake = wake_areas(coordinates,X,Y,wake_distance,U_direction,r_0)
#-------------If a cell lies in multiple wakes, apply the squares method highlighted in the paper
#-------------wake_loss also leaves the cell as 1 if the corresponding cell lies in no wakes. If in one wake, it leaves the cell 
#-------------as just being that value

    one_Z = wake_loss(list_of_jensens_factors,areas_in_wake,X,Y) #get array of (V/U), 
    Z = one_Z*V0                                                   #accompanying wind speeds, defecit x wind speed
    kw_power_list = get_power(Z,coordinates,U_direction,X,Y,r_0)    #get power of each turbine, by comparing wind speed with power curve
  
#With the next steps in the project, the above step will be replaced with a direct relationship between turbine distance and power (at different speeds? How does this look for different speeds)

    path = './angle_'+str(angle)+'_speed_'+str(V0)
    save(path,X,'X')
    save(path,Y,'Y')
    save(path,Z,'Z')
    

    
    print('\nData Saved!\n')

    #wind farm AEP
    AEP_data = AEP(kw_power_list,V0)

    import pandas as pd

    df = pd.DataFrame({'Turbine':turbine_name})
    df['Power'] = kw_power_list

    df.to_csv(path+'/power.csv')


    save(path,AEP_data,'power_generated')

    # print('-------------------Simulated Farm AEP------------------------\nTotal AEP: '+str(AEP_data[0])+' GWh')
    # print('Actual Revenue: £'+str(f"{AEP_data[1]:,}"))
    # print('From an available revenue of £'+str(f"{AEP_data[4]:,}"))
    # print('\n-------------------Farm Losses------------------------')
    # print('Whole Farm Loss: '+str(AEP_data[2])+' GWh')
    # print('Whole Farm Revenue Loss: £'+str(f"{AEP_data[3]:,}")+' \n')
    
    # plt.figure(1, figsize=(13,5))
    # plt.suptitle('Wind Farm Yield: '+str(AEP_data[0])+' GWh. Free energy: '+str(round(AEP_data[5],1))+' GWh. Loss: '+str(round(AEP_data[2]*100/AEP_data[5],1))+'%')

    # plt.subplot(1, 2, 1)
    # mycmap = cm.get_cmap('jet')
    # mycmap.set_under('w')   
    # plt.title('Wind Speed Flow Field')
    # plt.pcolormesh(X, Y, Z, shading='auto',cmap=mycmap)
    # plt.axis('scaled')
    # plt.xlabel('Local x-Direction /m')
    # plt.ylabel('Local y-Direction /m')
    # cbar = plt.colorbar()
    # cbar.set_label('Wind Speed /(m/s)')
    # turbine_no = range(1,len(coordinates)+1)

    # GWh_energy_list = list(map(lambda x: x * 8760/1000000, kw_power_list))
    
    # # plt.subplot(1, 2, 2)
    # # plt.xlabel('Turbine')
    # # plt.title('AEP Per Turbine')
    # # plt.ylabel('Yield /GWh')
    # # plt.grid(color='black', linestyle='-', linewidth=0.25)
    # # plt.bar( [str(int) for int in turbine_no],np.array(GWh_energy_list), color='green')

    # plt.figure(2)
    # plt.pcolormesh(X, Y, areas_in_wake, shading='auto')
    # # ax = Axes3D(plt.gcf())
    # # ax.plot_surface(X,Y,Z)


    plt.show()


def start():

    range_of_speeds = np.linspace(3,12,10)
    print(range_of_speeds)
    start=0
    step=5
    num=19

    angles_north_to_east=np.arange(0,num)*step+start  

    for i in range(0,len(angles_north_to_east)):
        for v in range(0,len(range_of_speeds)):
            main(angles_north_to_east[i],range_of_speeds[v])

if __name__ =='__main__':
    start()



# next steps
# cfd basic simulation of pylon + actuator disc
#----------21/12/20---------------------
#4 create algo, sanity case 

# 29/12/20
#added AEP calculations
#30/12/20
#linear distances using cross product
#plot yield for each turbine

#  revenue is predicted by simulating the losses of turbines
# power output is calculated from power curve & cut in/out wind speeds
# this


# next steps, calculate AEP of array formation for 16 directions for the avg speeds in those directions
# this simulates annual aep for a realistic real world scenario form real world data (when weightings for prevailaing wind are added)
