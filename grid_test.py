import matplotlib.pyplot as plt
import numpy as np
import time
import sys

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection


# https://stackoverflow.com/questions/25439243/find-the-area-between-two-curves-plotted-in-matplotlib-fill-between-area


r_0 = 63.2


# https://www.w3resource.com/python-exercises/basic/python-basic-1-exercise-40.php

# ((wakes[j])*np.sin(-U_direction_radians)+(r_0+alpha*wakes[j])*np.cos(-U_direction_radians))+y[i]

"0 degrees is coming from due North."
"+90 degrees means the wind is coming from due East, -90 from due West"
U_direction = float(sys.argv[1])
wind_direction = (U_direction+90) * np.pi / 180.


#  https://www.sciencedirect.com/science/article/pii/S0960148116309429?via%3Dihub


def equation_of_line1(x,turbine_coordinates, radius, direction):
    turbine_x = turbine_coordinates[0]
    turbine_y = turbine_coordinates[1]
    
    k_w = +0.04 
    line1_point2_x =  x*np.cos(-direction) - (radius+k_w*x)*np.sin(-direction) + turbine_x
    line1_point2_y =  x*np.sin(-direction) + (radius+k_w*x)*np.cos(-direction) + turbine_y

    return [line1_point2_x, line1_point2_y]

def equation_of_line2(x, turbine_coordinates, radius, direction):

    turbine_x = turbine_coordinates[0]
    turbine_y = turbine_coordinates[1]
    
    k_w = -0.04 
    line2_point2_x =  x*np.cos(-direction) - (-radius+k_w*x)*np.sin(-direction) + turbine_x
    line2_point2_y =  x*np.sin(-direction) + (-radius+k_w*x)*np.cos(-direction) + turbine_y
    
    return [line2_point2_x, line2_point2_y]





turbine_origin=[0, 0]

x = 1000

line1_point1 = turbine_origin
line1_point2 = equation_of_line1(x,turbine_origin, r_0, wind_direction)

line1_x = [line1_point1[0]+r_0, line1_point2[0]]
line1_y = [line1_point1[1], line1_point2[1]]



line2_point1 = turbine_origin
line2_point2 = equation_of_line2(x,turbine_origin, r_0, wind_direction)
print(line2_point2)
line2_x = [line2_point1[0]-r_0, line2_point2[0]]
line2_y = [line2_point1[1], line2_point2[1]]




plt.plot(line1_x, line1_y)
plt.plot(line2_x, line2_y)
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