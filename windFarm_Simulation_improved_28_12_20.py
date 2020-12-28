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

# https://stackoverflow.com/questions/25439243/find-the-area-between-two-curves-plotted-in-matplotlib-fill-between-area


#vestas v112 3.5MW


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
    #                                              s                                                                          ^  ^ these points rotate
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

def windspeed_probe(coordinates,angle,X,Y):
#-----------------------------------------------------------------------------------------------------------
#       Wind speed for each respective turbine needs to be measured from a slight upstream position
#       to the wind turbine coordinates. If incident wind speed is measured from the wind turbine center itself,
#       the down stream speed in the wake is recorded.
#-----------------------------------------------------------------------------------------------------------     
    theta = (angle) * np.pi / 180.

    x = float(coordinates[0])
    y = float(coordinates[1])
    # print(type(angle))
    #sign coefficients to ensure wakes start in right position, considering cos and sin  

    x_coord=  x+15*np.sin(theta)
    y_coord = y+15*np.cos(theta)
    
    turbine1_x_index=min(range(len(X[0])), key=lambda i: abs(X[0][i]-x_coord))   # returmns closest x and y 
    turbine1_y_index =min(range(len(X[0])), key=lambda i: abs(Y[i][0]-y_coord))

    print(x_coord)
    print(y_coord)
    print(turbine1_x_index)
    print(turbine1_y_index)
    print(X[turbine1_y_index][turbine1_x_index])
    print(Y[turbine1_y_index][turbine1_x_index])
    print('end\n\n\n')
    return [turbine1_x_index,turbine1_y_index]

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
    line1_xValues = [x1, line1_point2[0]] # array of x values for wake line 1
    line1_yValues = [y1, line1_point2[1]] # array of y values for line 1


    line2_point2 = []
    line2_point2 = equation_of_line2(wake_distance,[x2,y2], r_0, wind_direction) #get the distant point of the wake
    line2_xValues = [x2, line2_point2[0]] # array of x values for wake line 2
    line2_yValues = [y2, line2_point2[1]] # array of y values for wake line 2

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
    

def main(angle):

    # "0 degrees is coming from due North."
    # "+90 degrees means the wind is coming from due East, -90 from due West"
    U_direction = float(angle)

    r_0 = 56

    coordinates = []
    with open('turbines.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_file)
        for row in csv_reader:
            coordinates.append([row[1],row[2]])


    wake_distance = 6000 # in metres
    V0 = float(20)

    x = np.linspace(-1000,2000, 1000, endpoint = True) # x intervals
    y = np.linspace(-1000,1000,1000, endpoint = True) # y intervals
    X, Y = np.meshgrid(x,y)
    list_of_jensens_factors = []

    for i in range(0,len(coordinates)):

        coefficient_matrix = np.array(get_array_of_jensens_factor(wake_distance,coordinates[i],U_direction,r_0, X, Y))
        list_of_jensens_factors.append(coefficient_matrix)

    dummy = np.ones((X.shape[0],Y.shape[0]),dtype="float32")
    for i in range(0,len(list_of_jensens_factors)):
        dummy = np.multiply(dummy,list_of_jensens_factors[i])

    Z = dummy*V0

    probe_x, probe_y = windspeed_probe(coordinates[0],U_direction,X,Y)
    print('------------------\n\n')
    print('turbine 1 wind speed: '+ str(Z[probe_y][probe_x])+' meters per second\n\n')
    print('------------------')
    mycmap = cm.get_cmap('jet')
    mycmap.set_under('w')   
    plt.pcolor(X, Y, Z, shading='auto')
    plt.axis('scaled')
    plt.colorbar()
    plt.plot(X[probe_y][probe_x],Y[probe_y][probe_x], 'ro')
    

    # ax = Axes3D(plt.gcf())
    # ax.plot_surface(X,Y,Z)


    plt.show()



if __name__ =='__main__':
    main(sys.argv[1])



# next steps
# cfd basic simulation of pylon + actuator disc

#----------21/12/20---------------------
#1) feed in turbine x y coordinates by reading csv 
#2) create array of coeff for each array

#3) create ui showing power output due to each turbines wind speed
#4 create algo, sanity case 






#today 19/12/20
#) create function to check if a point lies within the wake
#) check that the function works, for different angles
#) create Z array for whole grid (grid of 1's) but applying 



