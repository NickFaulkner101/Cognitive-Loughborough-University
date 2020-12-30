import matplotlib.pyplot as plt
import numpy as np
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
    if (angle >= 0 and angle <= 90):
        A = -1
        B = 1 
        C = 1
        D = -1
    elif (angle > 90 and angle <= 180):
        A = -1
        B = 1 
        C = 1
        D = -1
    elif (angle < 0 and angle >= -90):
        A = -1
        B = 1 
        C = 1
        D = -1
    elif (angle < -90 and angle >= -180):
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

    x_coord=  x+10*np.sin(theta)
    y_coord = y+10*np.cos(theta)
    
    turbine1_x_index=min(range(len(X[0])), key=lambda i: abs(X[0][i]-x_coord))   # returmns closest x and y 
    turbine1_y_index =min(range(len(X[0])), key=lambda i: abs(Y[i][0]-y_coord))

    #turbine centre
    #pick 6 points along the turbine and calculate wind speed at each point
    #each point needs to be slightly upstream
    #then average the speeds
    #the below loop functions by working out the speed at 6 points radiating from the centre of the turbine
    speeds = []
    for i in range(1,4):
        #point 1 and point 2 are same distance either direction from wind turbine nacelle/centre
        x1 = x -(i/3)*r*np.cos(theta)
        x2 = x + (i/3)*r*np.cos(theta)
        y1 = y + (i/3)*r*np.sin(theta)
        y2 = y - (i/3)*r*np.sin(theta)

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

#----------- f2 returns the array of coefficients for a turbine------------------
#-----------x and y objects are just meshgrids for the coordinates------------------
def f2(X, Y, coords,origin):
    turbine_x = float(origin[0])
    turbine_y = float(origin[1])

    print('Calculating Turbine Array of Speed Deficiency Coefficients \n for turbine location: ' + str(origin))
    start = time.process_time()

    Ct = 0.8
    rd=23.5
    kw = 0.04 
    V0 = float(25)
    points = np.c_[X.ravel(), Y.ravel()]
    mask = Path(coords).contains_points(points).reshape(X.shape)
    result=np.ones((X.shape[0],Y.shape[0]),dtype="float32")

    for i in range (X.shape[0]):
        for j in range (Y.shape[0]):
            point = mask[j][i]
            if point == True:
                x = X[j][i]
                y = Y[j][i]
                distance = np.sqrt((x-turbine_x)**2+(y-turbine_y)**2) # should be linear distance from centre point?
                factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*distance/rd))**2))
                result[j,i] = factor
    print('Time for solution: '+str(time.process_time() - start)+' seconds\n')
    return result


def get_array_of_jensens_factor(wake_distance,turbine_origin,U_direction,r_0, X, Y):
    
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


    
    Z = f2(X, Y, coords,turbine_origin)
    # Z = f(X,Y, poly, turbine_origin)

    return Z

def power_curve(speed):
    power = 0.0676*speed**6 - 3.2433*speed**5 + 60.607*speed**4 - 565.82*speed**3 + 2830.8*speed**2 - 7083.5*speed + 6896.3
    return power

def get_power(Z,coordinates,U_direction,X,Y,r):
    power_generation_kw = []
    for i in range(0,len(coordinates)):
#-------------------------------------------------------------
#                   Estimated Wind Speed on Probe
        
        wind_speed = windspeed_probe(coordinates[i],U_direction,X,Y,r,Z)
        
#---------------------------------------------------------------
        if wind_speed < 3 or wind_speed > 25: #cut in and cut out
            power = 0
        elif wind_speed >= 3 and wind_speed <= 12.5: #power curve
            power = power_curve(wind_speed)
        elif wind_speed > 12.5 and wind_speed <= 25: # max power
            power = 3450
        power_generation_kw.append(power)
    return power_generation_kw



def AEP(power_kw,speed):
    #wind turbine should produce this amount in the wind
    num_turbines=len(power_kw)
    print('num-turbines'+str(num_turbines))
    power_freestream = power_curve(speed)
    total_free_power_kw = num_turbines*power_freestream
    free_GWh = total_free_power_kw*8760/1000000
    free_revenue=total_free_power_kw*8760*0.065

#------------------------total power of farm-------------------------
    total = sum(power_kw) #total power of farm at the windspeed, in kWh
    annual_GWh = total*8760/1000000  #AEP of wind farm considering wake effects
    revenue = total*8760*0.065

    loss = free_GWh-annual_GWh
    revenue_loss = free_revenue-revenue


    return [round(annual_GWh,2), round(revenue,2), round(loss,2),round(revenue_loss,2),round(free_revenue,2)]
    

def main(angle):

    # "0 degrees is coming from due North."
    # "+90 degrees means the wind is coming from due East, -90 from due West"
    U_direction = float(angle)
    V0 = float(10)
    print('----------Simulation Information')


    r_0 = 56

    coordinates = []
    with open('turbines.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_file)
        for row in csv_reader:
            coordinates.append([row[1],row[2]])


    wake_distance = 6000 # in metres

    x = np.linspace(-1000,3000, 1000, endpoint = True) # x intervals
    y = np.linspace(-2000,2000,1000, endpoint = True) # y intervals
    X, Y = np.meshgrid(x,y)
    list_of_jensens_factors = []

    for i in range(0,len(coordinates)):

        coefficient_matrix = np.array(get_array_of_jensens_factor(wake_distance,coordinates[i],U_direction,r_0, X, Y))
        list_of_jensens_factors.append(coefficient_matrix)

    jensens_factors = np.ones((X.shape[0],Y.shape[0]),dtype="float32")
    for i in range(0,len(list_of_jensens_factors)):
        jensens_factors = np.multiply(jensens_factors,list_of_jensens_factors[i])

    Z = jensens_factors*V0

    kw_power_list = get_power(Z,coordinates,U_direction,X,Y,r_0)

    #wind farm AEP

    AEP_data = AEP(kw_power_list,V0)

    print('-------------------Simulated Farm AEP------------------------\n\nTotal AEP: '+str(AEP_data[0])+' GWh')
    print('\nActual Revenue: £'+str(f"{AEP_data[1]:,}"))
    print('\nFrom an available revenue of £'+str(f"{AEP_data[4]:,}"))
    print('\n-------------------Farm Losses------------------------')
    print('\nWhole Farm Loss: '+str(AEP_data[2])+' GWh')
    print('\nWhole Farm Revenue Loss: £'+str(f"{AEP_data[3]:,}")+' \n')
    

    mycmap = cm.get_cmap('jet')
    mycmap.set_under('w')   
    plt.pcolor(X, Y, Z, shading='auto')
    plt.axis('scaled')
    plt.colorbar()
    # plt.plot(X[probe_y][probe_x],Y[probe_y][probe_x], 'ro')
    

    # ax = Axes3D(plt.gcf())
    # ax.plot_surface(X,Y,Z)


    plt.show()



if __name__ =='__main__':
    main(sys.argv[1])



# next steps
# cfd basic simulation of pylon + actuator disc
#----------21/12/20---------------------
#4 create algo, sanity case 

# 29/12/20
#added AEP calculations


#  revenue is predicted by simulating the losses of turbines
# power output is calculated from power curve & cut in/out wind speeds
# this


# next steps, calculate AEP of array formation for 16 directions for the avg speeds in those directions
# this simulates annual aep for a realistic real world scenario form real world data (when weightings for prevailaing wind are added)