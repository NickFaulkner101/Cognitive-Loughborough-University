import matplotlib.pyplot as plt
import numpy as np
import time
import sys

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection


# https://stackoverflow.com/questions/25439243/find-the-area-between-two-curves-plotted-in-matplotlib-fill-between-area


r_0 = 56
#vestas v112 3.5MW

# https://www.w3resource.com/python-exercises/basic/python-basic-1-exercise-40.php

# ((wakes[j])*np.sin(-U_direction_radians)+(r_0+alpha*wakes[j])*np.cos(-U_direction_radians))+y[i]

"0 degrees is coming from due North."
"+90 degrees means the wind is coming from due East, -90 from due West"
U_direction = float(sys.argv[1])
wind_direction = (U_direction+90) * np.pi / 180. # plus 90 to align with 0 degrees as north


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



turbine_origin=[0, 0]

x = 1000


# -----------1 refers to point 1, 2 refers to point 2, the two points represent either side of the turbine swept area
x1, x2, y1, y2 = wake_start_points(turbine_origin,U_direction,r_0)







line1_point2 = equation_of_line1(x,[x1,y1], r_0, wind_direction) #get the distant point of the wake
line1_xValues = [x1, line1_point2[0]] # array of x values for wake line 1
line1_yValues = [y1, line1_point2[1]] # array of y values for line 1


line2_point2 = []
line2_point2 = equation_of_line2(x,[x2,y2], r_0, wind_direction) #get the distant point of the wake
line2_xValues = [x2, line2_point2[0]] # array of x values for wake line 2
line2_yValues = [y2, line2_point2[1]] # array of y values for wake line 2

plt.plot(line1_xValues, line1_yValues, 'b-')
plt.plot(line2_xValues, line2_yValues, 'r-')
plt.axis('scaled')
plt.show()












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