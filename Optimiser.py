import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot
import scipy
import sys
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
import operator

# for reference to equations
#  https://www.tandfonline.com/doi/pdf/10.1080/22348972.2015.1109793#:~:text=Jensen%20model%20is%20one%20of,the%20wind%20speed%20time%20delay.
# https://onlinelibrary.wiley.com/doi/full/10.1002/we.1863
## list relationship of wind speed incident on wind turbine


def Jensen(combined, speed):
    V0 = float(speed)
    Ct = 0.8
    rd=23.5
    kw = 0.04 
    # diameter = 2*rd
    # ro is rotor radius
    # r radius of wake
    # rudimentary jensen's model for far wake regions, 6-8D off shore
    # v1 = V0 +V0*(math.sqrt(1-Ct)-1)*(rd/r)**2
    velocities = [] # velocities at wake radii up to 10D
    for x in range(0,1000):
        v_next = V0*(1-((1-math.sqrt(1-Ct))/(1+(kw*x/rd))**2))
        velocities.append(v_next)
    # 6.5p per kwh
    power_yield = []
    for i in range(0, len(velocities)):
        A= math.pi*rd**2
        ro=1.223
        Cp = 0.4
        x = velocities[i]
        power_kw = (-0.0199)*(x)**6 + (0.6015)*(x)**5  + (-7.1259)*(x)**4 + (43.755)*(x)**3 + (- 111.34)*x**2 + (105.72)*x -28.696
        power_yield.append(power_kw)

    power_at_half_km = power_yield[500]
    wind_speed = velocities[500]
    print('\n Jensen Model\n3.5MW turbine 0.5km from start of wake:')
    print('wind speed: '+ str(round(wind_speed, 2)) + ' m/s')
    print('wind power: '+str(round(power_at_half_km, 0)) + ' kW')
    print('AEP (Annual Energy Production): ' + str(round(power_at_half_km*8760/1000000, 0))+ ' GWh')
    print('revenue per year: £'+str(round(power_at_half_km*8760*0.065, 2)))

    #inaccurate polynomial below certain velocity/distance

    if combined == 'Y':
        return power_yield
    if combined == 'N':
        AEP_calculations(power_yield)

def Frandsen(combined,speed):
    V0 = float(speed)
    Ct = 0.8
    rd=23.5
    
    diameter = 2*rd
    beta = 0.5*((1+math.sqrt(1+Ct))/math.sqrt(1-Ct))
    # ro is rotor radius
    # r radius of wake
    # rudimentary jensen's model for far wake regions, 6-8D off shore
    # v1 = V0 +V0*(math.sqrt(1-Ct)-1)*(rd/r)**2
    velocities = [] # velocities at wake radii up to 10D
    for x in range(0,1000):
        Dx = (1+2*0.04*(x/(diameter)))*math.sqrt(beta)*diameter
        v_next = V0*(0.5 + 0.5*math.sqrt(1-2*Ct*(diameter/Dx)**2))
        velocities.append(v_next)
        
    power_yield = []
    for i in range(0, len(velocities)):
        x = velocities[i]
        power_kw = (-0.0199)*(x)**6 + (0.6015)*(x)**5  + (-7.1259)*(x)**4 + (43.755)*(x)**3 + (- 111.34)*x**2 + (105.72)*x -28.696
        power_yield.append(power_kw)

    power_at_half_km = power_yield[500]
    wind_speed = velocities[500]
    print('\n Frandsen Model\n 3.5MW turbine 0.5km from start of wake:')
    print('wind speed: '+ str(round(wind_speed, 2)) + ' m/s')
    print('wind power: '+str(round(power_at_half_km, 0)) + ' kW')
    print('AEP (Annual Energy Production): ' + str(round(power_at_half_km*8760/1000000, 0))+ ' GWh')
    print('revenue per year: £'+str(round(power_at_half_km*8760*0.065, 2)))

    if combined == 'Y':
        return power_yield
    if combined == 'N':
        AEP_calculations(power_yield)
 
def AEP_calculations(power):

    plt.plot(power)
    plt.ylabel('Power Yield/ kWh ')
    plt.show()

def optimiser(speed):
    Jensen_Yield = Jensen('Y',speed)
    cost_matrix = []
    # maximise the profit over the year 
    # for each distance, compute the AEP profit - misc one-off costs

    #

    for i in range(0,1000):
        annual_profit = Jensen_Yield[i]*8760*0.065 #profit at each metre away from lead turbine
        length_cost = 100*i #per metre cost per year
        cost_matrix.append(annual_profit-length_cost) #for each distance, the profit from energy - cost from power losses is calculated
    return cost_matrix

    #optimiser


def model(speed):

    plt.figure(3, figsize=(13,5)) #plot figure of optimal values
    total_profit = np.array(optimiser(speed))
    plt.plot(total_profit)
    max_index, max_value = max(enumerate(total_profit), key=operator.itemgetter(1))
    plt.annotate(text='AEP Revenue: £'+str(round(max_value,2)), xy=(max_index, max_value-5000), xycoords='data')
    plt.annotate(text='Distance from lead turbine : '+str(max_index)+' m', xy=(max_index, max_value-7500), xycoords='data')
    plt.xlabel('Distance /m')
    plt.grid(color='b', linestyle='-', linewidth=0.1)
    plt.ylabel('Revenue /£')
    plt.title('2nd Turbine Optimal Location')
    
    
    plt.figure(1, figsize=(13,5))
    
    Frandsen_Yield = Frandsen('Y',speed) #get frandsen yields for each distance
    Jensen_Yield = Jensen('Y',speed) #get jensen yields for each distance
    df1 = pd.DataFrame({'Frandsen_Yield':Frandsen_Yield}) #load yield data into dataframes
    df2 = pd.DataFrame({'Jensen_Yield':Jensen_Yield}) #load yield data into dataframes
    result = pd.concat([df1, df2], axis=1, sort=False)  #combine into single dataframe for plotting
  
    

    plt.subplot(1, 2, 1)
    plt.plot(result['Frandsen_Yield'], label='Frandsen Yield')
    plt.plot(result['Jensen_Yield'], label='Jensen Yield')
    plt.title('Frandsen_Yield vs Jensen Power vs Distance from Upstream Wind Turbine')
    plt.xlabel('Distance /m')
    plt.ylabel('Power /kW')
    plt.legend(loc='upper left', fontsize=8)

    plt.subplot(1, 2, 2)
    data = [ Frandsen_Yield[500]*8760/1000000, Jensen_Yield[500]*8760/1000000]
    plt.xlabel('Method')
    plt.title('Total Annual Energy Production Comparison 0.5km')
    plt.ylabel('Yield /GWh')
    plt.bar([0, 1], data, color='green')
    plt.xticks([0, 1], ['Frandsen', 'Jensen'])
    
    plt.figure(2)
    ax = Axes3D(plt.gcf())
    U_direction = -90.
    U_direction_radians = (U_direction+90) * np.pi / 180.
    x = np.array([0,max_index], dtype="object") #x coordinates of the turbine
    y = np.array([250, 250], dtype="object") #y coordinates of the turbine
    z = np.array([69, 69]) #hub height of each turbine
    # wakes = np.linspace(0, 1000, num=101)
    ax.set_box_aspect([4,2,1],  zoom=1.2)
    r_0 = 63.2
    alpha = 0.10033467208545055
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], c = 'r', s=np.pi*r_0**2, marker='2')
        xtemp = (x[i], x[i])
        ytemp = (y[i], y[i])
        ztemp = (0, z[i])
        ax.plot(xtemp, ytemp, ztemp, zdir='z', c='r', linewidth = 5.0)
    ax.set_xlim([-100, 1000])
    ax.set_ylim([0, 500])  
    ax.set_zlim([0, 120])
    wake = pd.DataFrame([])



    intercept = 250 #location of lead turbine (centre of width of grid)        
    gradient1 = (intercept+r_0-(intercept+intercept))/(-1000) #adjust wake for diameter of blade up to 1k
    gradient2 = (intercept-r_0-(intercept-intercept))/(-1000) #adjust wake for diameter of blade up to 1k
    for i in range(0,1000):
        wake = wake.append(pd.DataFrame({'positive': gradient1*i+(intercept+63.2), 'negative': gradient2*i+(intercept-63.2)}, index=[0]), ignore_index=True)
    ax.plot(wake.index, wake['positive'],z[0], c='b')
    ax.plot(wake.index, wake['negative'],z[0], c='b')
    ax.plot(wake.index, wake['positive'],0, c='b')
    ax.plot(wake.index, wake['negative'],0, c='b')
    ax.set_xlabel('Distance from Turbine 1 /m')
    ax.set_ylabel('Horizontal Distance /m')
    ax.set_title('subplot 1')
    ax.set_xlabel('Distance from Turbine 1 /m')
    ax.set_ylabel('Horizontal Distance /m')
    ax.set_title('subplot 1')

    X = [0, 999, 0, 999]
    Y = [float(intercept + r_0), wake['positive'][999], float(intercept - r_0), wake['negative'][999]]
    Z = [0,0,0,0]


    



   
    
    print('values based on an average annual wind speed of '+str(speed)+' m/s')
    print('optimal distance ='+str(max_index)+'m')
    plt.show()

def optimisationFunction():
    V0 = 5 #upstream average wind speed in english channel (guess)
    rd=23.5 #rotor diameter
    kw = 0.04 #expansion factor of wake
    Ct = 0.8
    revenues = []
    for i in range(0,1000):
        x = V0*(1-((1-math.sqrt(1-Ct))/(1+(kw*x/rd))**2)) #example from jensen
        # 6th order polynomial derived from Vestas V112 3.5MW 
        power_kw = (-0.0199)*(x)**6 + (0.6015)*(x)**5  + (-7.1259)*(x)**4 + (43.755)*(x)**3 + (- 111.34)*x**2 + (105.72)*x -28.696
        revenue = power_kw*8760*0.065 - 100*i # subtract 'power losses' per metre away from turbine
        revenues.append(revenue)
    plt.plot(revenues)
    plt.ylabel('Revenues/ £ ')
    plt.show()

def main(simulation_type):
    
    windspeed = 5 #m/s
    if simulation_type == 'jensen':
        Jensen('N',windspeed)
    if simulation_type == 'frandsen':
        Frandsen('N',windspeed)
    if simulation_type == 'test':
        model(windspeed)



if __name__ =='__main__':
    main(sys.argv[1])




#currently finds optimal location
# identify cost of power losses per metre
# i.e. 100 metres loses 4kw, which is £4*8760*0.65 per year if that makes sense