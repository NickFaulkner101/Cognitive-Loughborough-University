import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import PolyCollection
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon






# https://stackoverflow.com/questions/25439243/find-the-area-between-two-curves-plotted-in-matplotlib-fill-between-area


#vestas v112 3.5MW


#  https://www.sciencedirect.com/science/article/pii/S0960148116309429?via%3Dihub


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

    x = coordinates[0]
    y = coordinates[1]

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

def f(x_object, y_object, poly, origin):
    turbine_x = origin[0]
    turbine_y = origin[1]
    V0 = float(25) #incumbent wind speed
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
    
    # your code here    
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
    poly = Polygon(coords)
    # plt.plot(*poly.exterior.xy)
    #-------------Check If Point is in polygon--------------
    p1 = Point(65, 110)
    # print(p1.within(poly))



    # plt.plot(line1_xValues, line1_yValues, 'b-')
    # plt.plot(line2_xValues, line2_yValues, 'r-')
    # plt.axis('scaled')


    

    Z = f(X,Y, poly, turbine_origin)
    return Z
    

def main(angle):

    # "0 degrees is coming from due North."
    # "+90 degrees means the wind is coming from due East, -90 from due West"
    U_direction = float(angle)

    r_0 = 56
    turbine1_origin=[0, 0]
    turbine2_origin=[500, 500]
    turbine3_origin=[500, -500]
    turbine4_origin=[-500, 500]
    turbine5_origin=[-500, -500]
    wake_distance = 5000 # in metres
    V0 = float(25)

    x = np.linspace(-5000,5000, 500, endpoint = True) # x intervals
    y = np.linspace(-5000,5000,500, endpoint = True) # y intervals
    X, Y = np.meshgrid(x,y)

    A = np.array(get_array_of_jensens_factor(wake_distance,turbine1_origin,U_direction,r_0, X, Y))

    B = np.array(get_array_of_jensens_factor(wake_distance,turbine2_origin,U_direction,r_0, X, Y))

    C = np.array(get_array_of_jensens_factor(wake_distance,turbine3_origin,U_direction,r_0, X, Y))

    D = np.array(get_array_of_jensens_factor(wake_distance,turbine4_origin,U_direction,r_0, X, Y))

    E = np.array(get_array_of_jensens_factor(wake_distance,turbine5_origin,U_direction,r_0, X, Y))
    
    AandB = np.multiply(A,B)

    CandD = np.multiply(C,D)

    CandDandE = np.multiply(CandD,E)

    Z = np.multiply(AandB, CandDandE)
    Z = Z*V0

    print(Z)

    

    plt.figure(1, figsize=(13,5))
    plt.pcolor(X, Y, Z, shading='auto')
    plt.colorbar()
    plt.axis('scaled')
    plt.show()



if __name__ =='__main__':
    main(sys.argv[1])














# z is wind speed, function of distance from turbine (in this case at 0,0)
#  point:

#1) how do you iterate through Z, which is the field of velocities, to add another turbine
# ^ create Z array for each turbine, then do element by element multiplication for the whole grid for each turbine
# The z arrays calculated for each turbine represents the wind speed deficiency i.e. jensen speed, for each wake as a function of distance

#today 19/12/20
#) create function to check if a point lies within the wake
#) check that the function works, for different angles
#) create Z array for whole grid (grid of 1's) but applying 






# tomorrow
#add auto equation of lines from points into the Z function limits, i.e. find gradient and intercept of wake lines
# warning: these may change as line rotates, i.e. sign changes. How to deal with etc



# ax = Axes3D(plt.gcf())
# ax.plot_surface(X,Y,Z)
# plt.show()



















# p = np.pi
# pi = np.pi
# x = np.linspace(0,8*p, 100, endpoint = True) # x intervals
# y = np.linspace(0,1,10, endpoint = True) # y intervals

# def f(x, y):
#     return y * np.np.sin(x) # returns a z value as a function of each x y pair

# X, Y = np.meshgrid(x,y) 
# # X is the x values for each row of Y
# # Y is the y values for each column of X
# Z = f(X, Y) # iteratively produces a z for each xy pair 
# print(X.shape) #(10,100)
# print(Y.shape) #(10,100)
# print(Z.shape) #(10,100)

# plt.figure(1, figsize=(13,5))
# plt.pcolor(X, Y, Z)



# plt.figure(2, figsize=(13,5))
# ax = Axes3D(plt.gcf())

# ax.plot_surface(X, Y, Z)

# plt.show()






# test_x = 0 + 50*np.np.cos(-U_direction_radians)-(r_0+alpha*50)*np.np.sin(-U_direction_radians)
# test_y = 250 + 50*np.np.sin(-U_direction_radians)+(r_0+alpha*50)*np.np.cos(-U_direction_radians)


# ax.plot([test_x], [test_y], [0], c = 'r', marker='1')