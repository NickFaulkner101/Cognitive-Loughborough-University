import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from geopy.distance import geodesic
from sklearn import linear_model, datasets
import math
from scipy.interpolate import make_interp_spline, BSpline



#plot same as earlier, for each turbine ararngement

def main():
    # get_ransac()
    # get_graphs_RANSAC_prediction()
    get_all_data_and_polyfit()


def return_power_value(speed,powercurve_windspeed_new,power_smooth):
    index_power = min(range(len(powercurve_windspeed_new)), key=lambda i: abs(powercurve_windspeed_new[i]-speed))
    power_value = power_smooth[index_power]
    return power_value


def get_ct_value(speed):

    #modified values
    windspeed_values= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
    windspeed_values_new = np.linspace(windspeed_values.min(), windspeed_values.max(), 100)  
    # ctvalues = [0.691,0.685,0.665,0.639,0.640,0.640,0.634,0.627,0.620,0.612,0.603,0.601,0.696,0.684,0.53,0.698,0.627,0.546,0.469,0.403,0.391,0.350,0.3,0.355,0.219,0.298,0.280,0.264,0.250,0.238,0.227,0.217,0.208,0.200,0.193,0.187,0.181,0.177,0.172,0.168,0.164,0.160,0.157,0.154,0.151]

    # ctvalues = 0.6*np.ones(len(windspeed_values)) #ct values that are constant

    # #original actual turbine values
    # # windspeed_values= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
    # # windspeed_values_new = np.linspace(windspeed_values.min(), windspeed_values.max(), 100)  
    ctvalues = [0.891,0.885,0.865,0.839,0.840,0.840,0.834,0.827,0.820,0.812,0.803,0.801,0.796,0.784,0.753,0.698,0.627,0.546,0.469,0.403,0.351,0.310,0.275,0.245,0.219,0.198,0.180,0.164,0.150,0.138,0.127,0.117,0.108,0.100,0.093,0.087,0.081,0.077,0.072,0.068,0.064,0.060,0.057,0.054,0.051]


    spl = make_interp_spline(windspeed_values, ctvalues, k=1)  # type: BSpline
    ct_values_smoothed = spl(windspeed_values_new)

    index_speed = min(range(len(windspeed_values_new)), key=lambda i: abs(windspeed_values_new[i]-speed))
    returned_ct = ct_values_smoothed[index_speed]

    return returned_ct


def get_ct_value_enhanced(speed):

    #modified values
    windspeed_values= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
    windspeed_values_new = np.linspace(windspeed_values.min(), windspeed_values.max(), 100)  
    ctvalues = [0.691,0.685,0.665,0.639,0.640,0.640,0.634,0.627,0.620,0.612,0.603,0.601,0.696,0.684,0.53,0.698,0.627,0.546,0.469,0.403,0.391,0.350,0.3,0.355,0.219,0.298,0.280,0.264,0.250,0.238,0.227,0.217,0.208,0.200,0.193,0.187,0.181,0.177,0.172,0.168,0.164,0.160,0.157,0.154,0.151]

    # ctvalues = 0.6*np.ones(len(windspeed_values)) #ct values that are constant

    # #original actual turbine values
    # # windspeed_values= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25])
    # # windspeed_values_new = np.linspace(windspeed_values.min(), windspeed_values.max(), 100)  
    # ctvalues = [0.891,0.885,0.865,0.839,0.840,0.840,0.834,0.827,0.820,0.812,0.803,0.801,0.796,0.784,0.753,0.698,0.627,0.546,0.469,0.403,0.351,0.310,0.275,0.245,0.219,0.198,0.180,0.164,0.150,0.138,0.127,0.117,0.108,0.100,0.093,0.087,0.081,0.077,0.072,0.068,0.064,0.060,0.057,0.054,0.051]


    spl = make_interp_spline(windspeed_values, ctvalues, k=1)  # type: BSpline
    ct_values_smoothed = spl(windspeed_values_new)

    index_speed = min(range(len(windspeed_values_new)), key=lambda i: abs(windspeed_values_new[i]-speed))
    returned_ct = ct_values_smoothed[index_speed]

    return returned_ct


def residuals(x,y,Turbine_A,Turbine_B,name):
    standard_deviation = np.std(Turbine_B)
    residuals = []
    square_differences = []
    squared_difference_to_mean = []
    Turb_B_mean = np.mean(Turbine_B)
    for i in range(0,len(Turbine_A)):
        if i % 100 == 0:
            print(str(i) + '/' + str(len(Turbine_A))+' Calculating Residuals')

        #corresponding downstream value
        downstream_value = Turbine_B[i]
        #corresponding predicted value
        upstream_value = Turbine_A[i]
        #index to find the upstream value that corresponds
        index = min(range(len(x)), key=lambda i: abs(x[i]-upstream_value))
        corresponding_model_prediction = y[index]
        normalised_difference = float(downstream_value - corresponding_model_prediction)/standard_deviation

        square_difference = (downstream_value - corresponding_model_prediction)**2
        square_differences.append(square_difference)
        squared_difference_mean = (downstream_value - Turb_B_mean )**2
        squared_difference_to_mean.append(squared_difference_mean)

        residuals.append(float(normalised_difference))
        

    square_differences = np.array(square_differences)
    squared_difference_to_mean = np.array(squared_difference_to_mean)


    sum_square_difference = np.sum(square_differences)
    sum_square_difference_to_mean = np.sum(squared_difference_to_mean)
    print(sum_square_difference_to_mean)
    print(sum_square_difference)
    r_2 = 1 - sum_square_difference/sum_square_difference_to_mean
    print('R Squared = ' + str(r_2))
    RMSE = np.sqrt(sum_square_difference/len(Turbine_A))
    print(RMSE)
    print('RMSE^^^^')
    print(RMSE/standard_deviation)
    print('Normalised RMSE')

    import scipy.stats as stats
    print(np.array(residuals))
    plt.figure(1000)
    stats.probplot(residuals, dist="norm", plot=plt,rvalue=True)

    plt.title("Q-Q Plot of Residuals from Jensen Fit of " +name+ " Data",fontsize=16)
    plt.xlabel('Theoretical Quantiles', fontsize=16)
    plt.ylabel('Ordered Values', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.show()


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


def get_general_ransac():
    return

def get_smoothed_jensen_line(distances):

    distance  = np.average(distances)
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


    x_smoothed = np.linspace(x.min(), x.max(), 5000)
    spl = make_interp_spline(x, single_wake_data, k=1)
    jensen_smoothed = spl(x_smoothed)



    return [x_smoothed,jensen_smoothed]


def get_smoothed_jensen_line_Enhanced_Ct(distances):

    distance  = np.average(distances)
    x = np.linspace(0,25,26) #up to max wind speed

    

    jensen_speed = []
    single_wake_data = []

    for i in range(0,len(x)):
        rd=56
        kw = 0.04
        turbine_distance = float(distance)
        Ct = get_ct_value_enhanced(i)
        factor = (1-((1-math.sqrt(1-Ct))/(1+(kw*turbine_distance/rd))**2))
        jensen_speed.append(factor)
        single_wake_data.append(factor*i)


    x_smoothed = np.linspace(x.min(), x.max(), 1000)
    spl = make_interp_spline(x, single_wake_data, k=1)
    jensen_smoothed = spl(x_smoothed)



    return [x_smoothed,jensen_smoothed]

       
def get_ransac():
    powercurve_windspeed= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5])
    powercurve_windspeed_new = np.linspace(powercurve_windspeed.min(), powercurve_windspeed.max(), 3000)  
    powercurve_power = [7,53,123,208,309,427,567,732,927,1149,1401,1688,2006,2348,2693,3011,3252,3388,3436,3448,3450]

    spl = make_interp_spline(powercurve_windspeed, powercurve_power, k=1)  # type: BSpline
    #powercurve load in
    power_smooth = spl(powercurve_windspeed_new)

    
    turbine_list = csv.reader(open('turbine_list_46.txt', "r"), delimiter=",")
    next(turbine_list)

    Turbine_As_power = []
    Turbine_Bs_power = []

    distances = []

    for row in turbine_list:

        lead = row[0]
        behind = row[1]

        distance = get_distances(lead,behind)
        distances.append(distance)

        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')

        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = Turbine_A_power.reshape(-1,1)
        Turbine_B_power = Turbine_B_power.reshape(-1,1)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)
        
        ransac = linear_model.RANSACRegressor()
        ransac.fit(Turbine_A_power, Turbine_B_power)

        print('RANSAC estimator coefficient: ' + str(ransac.estimator_.coef_))

        line_X = np.arange(Turbine_A_power.min(), Turbine_A_power.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)


      


    turbine_list = csv.reader(open('turbine_list_226.txt', "r"), delimiter=",")
    next(turbine_list)


    for row in turbine_list:

        lead = row[0]
        behind = row[1]


        distance = get_distances(lead,behind)
        distances.append(distance)

        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        


        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = Turbine_A_power.reshape(-1,1)
        Turbine_B_power = Turbine_B_power.reshape(-1,1)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)
        
        ransac = linear_model.RANSACRegressor()
        ransac.fit(Turbine_A_power, Turbine_B_power)

        print('RANSAC estimator coefficient: ' + str(ransac.estimator_.coef_))

        line_X = np.arange(Turbine_A_power.min(), Turbine_A_power.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        


    turbine_list = csv.reader(open('turbine_list_106.txt', "r"), delimiter=",")
    next(turbine_list)


    for row in turbine_list:

        lead = row[0]
        behind = row[1]


        distance = get_distances(lead,behind)
        distances.append(distance)


        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        

        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = Turbine_A_power.reshape(-1,1)
        Turbine_B_power = Turbine_B_power.reshape(-1,1)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)
        
        ransac = linear_model.RANSACRegressor()
        ransac.fit(Turbine_A_power, Turbine_B_power)

        print('RANSAC estimator coefficient: ' + str(ransac.estimator_.coef_))

        line_X = np.arange(Turbine_A_power.min(), Turbine_A_power.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

       



    jensen_line = get_smoothed_jensen_line(distances)


    Turbine_As_power = np.asarray(Turbine_As_power).reshape(-1,1)
    Turbine_Bs_power = np.asarray(Turbine_Bs_power).reshape(-1,1)

    ransac = linear_model.RANSACRegressor()
    ransac.fit(Turbine_As_power, Turbine_Bs_power)

    general_ransac_x = np.arange(Turbine_As_power.min(), Turbine_As_power.max())[:, np.newaxis]
    general_ransac_y = ransac.predict(general_ransac_x)

    x_1 = general_ransac_x[0]
    x_2 = general_ransac_x[len(general_ransac_x)-1]

    y_1 = general_ransac_y[0]
    y_2 = general_ransac_y[len(general_ransac_y)-1]

    # overall_ransac_points_x = [general_ransac_x[0],general_ransac_x[len(general_ransac_x)]]
    # overall_ransac_points_y = [general_ransac_y[0],general_ransac_y[len(general_ransac_x)]]

    jensen_x_axis = []
    jensen_y_axis = []
    for i in range(0,len(jensen_line[0])):
        power = return_power_value(jensen_line[0][i],powercurve_windspeed_new,power_smooth)
        jensen_x_axis.append(power)

        power = return_power_value(jensen_line[1][i],powercurve_windspeed_new,power_smooth)
        jensen_y_axis.append(power)


    print(str(ransac.estimator_.coef_))
    plt.figure(1)
    plt.plot([x_1, x_2],[y_1, y_2],color='red', linewidth=2,label='RANSAC regressor')
    plt.plot(jensen_x_axis,jensen_y_axis,color='green', linewidth=1,label='Jensen Prediction')
    plt.legend(loc="upper left")
    plt.xlabel('Turbine A Power (kW)')
    plt.ylabel('Turbine B Power (kW)')
    plt.title('Jensen-Predicted Power versus RANSAC of Turbine Data ')
    plt.grid()
    plt.show()


def get_graphs_RANSAC_prediction():

    powercurve_windspeed= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5])
    powercurve_windspeed_new = np.linspace(powercurve_windspeed.min(), powercurve_windspeed.max(), 3000)  
    powercurve_power = [7,53,123,208,309,427,567,732,927,1149,1401,1688,2006,2348,2693,3011,3252,3388,3436,3448,3450]

    spl = make_interp_spline(powercurve_windspeed, powercurve_power, k=1)  # type: BSpline
    #powercurve load in
    power_smooth = spl(powercurve_windspeed_new)

    
    turbine_list = csv.reader(open('turbine_list_46.txt', "r"), delimiter=",")
    next(turbine_list)

    Turbine_As_power = []
    Turbine_Bs_power = []

    i = 1
    distances = []

    for row in turbine_list:
        plt.figure(i)

        lead = row[0]
        behind = row[1]

        distance = get_distances(lead,behind)
        distances.append(distance)

        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df[behind+'_Grd_Prod_Pwr_Avg'],marker='x',s=1,color="blue", label='Turbine Actual Data')
        # plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

        plt.title('Turbine A Actual Power Versus Turbine B Predicted Power ' + lead + ' ' + behind)
        plt.xlabel('Turbine A Power (kW)')
        plt.ylabel('Turbine B Predicted Power (kW)')
        plt.grid()

        # plt.scatter(df['power_inferred_'+lead+'_windspeed'],df['power_inferred_'+behind+'_windspeed'])
        # plt.scatter(df['power_inferred_'+lead+'_windspeed'],df['jensen_'+behind+'_windspeed'])


        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = Turbine_A_power.reshape(-1,1)
        Turbine_B_power = Turbine_B_power.reshape(-1,1)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)
        
        ransac = linear_model.RANSACRegressor()
        ransac.fit(Turbine_A_power, Turbine_B_power)

        print('RANSAC estimator coefficient: ' + str(ransac.estimator_.coef_))

        line_X = np.arange(Turbine_A_power.min(), Turbine_A_power.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        plt.plot(line_X,line_y_ransac,color='red', linewidth=2,label='RANSAC regressor')
        plt.legend(loc="upper left")

        
        i = i+1


    turbine_list = csv.reader(open('turbine_list_226.txt', "r"), delimiter=",")
    next(turbine_list)

    i = 50

    for row in turbine_list:
        plt.figure(i)

        lead = row[0]
        behind = row[1]


        distance = get_distances(lead,behind)
        distances.append(distance)

        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df[behind+'_Grd_Prod_Pwr_Avg'],marker='x',s=1,color="blue", label='Turbine Actual Data')
        # plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

        plt.title('Turbine A Actual Power Versus Turbine B Predicted Power ' + lead + ' ' + behind)
        plt.xlabel('Turbine A Power (kW)')
        plt.ylabel('Turbine B Predicted Power (kW)')
        plt.grid()

        # plt.scatter(df['power_inferred_'+lead+'_windspeed'],df['power_inferred_'+behind+'_windspeed'])
        # plt.scatter(df['power_inferred_'+lead+'_windspeed'],df['jensen_'+behind+'_windspeed'])


        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = Turbine_A_power.reshape(-1,1)
        Turbine_B_power = Turbine_B_power.reshape(-1,1)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)
        
        ransac = linear_model.RANSACRegressor()
        ransac.fit(Turbine_A_power, Turbine_B_power)

        print('RANSAC estimator coefficient: ' + str(ransac.estimator_.coef_))

        line_X = np.arange(Turbine_A_power.min(), Turbine_A_power.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        plt.plot(line_X,line_y_ransac,color='red', linewidth=2,label='RANSAC regressor')
        plt.legend(loc="upper left")

     
        i = i+1


    turbine_list = csv.reader(open('turbine_list_106.txt', "r"), delimiter=",")
    next(turbine_list)

    i = 100

    for row in turbine_list:
        plt.figure(i)

        lead = row[0]
        behind = row[1]


        distance = get_distances(lead,behind)
        distances.append(distance)


        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df[behind+'_Grd_Prod_Pwr_Avg'],marker='x',s=1,color="blue", label='Turbine Actual Data')
        # plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

        plt.title('Turbine A Actual Power Versus Turbine B Predicted Power ' + lead + ' ' + behind)
        plt.xlabel('Turbine A Power (kW)')
        plt.ylabel('Turbine B Predicted Power (kW)')
        plt.grid()

        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = Turbine_A_power.reshape(-1,1)
        Turbine_B_power = Turbine_B_power.reshape(-1,1)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)
        
        ransac = linear_model.RANSACRegressor()
        ransac.fit(Turbine_A_power, Turbine_B_power)

        print('RANSAC estimator coefficient: ' + str(ransac.estimator_.coef_))

        line_X = np.arange(Turbine_A_power.min(), Turbine_A_power.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        plt.plot(line_X,line_y_ransac,color='red', linewidth=2,label='RANSAC regressor')
        plt.legend(loc="upper left")

        
        i = i+1






















    # jensen_line = get_smoothed_jensen_line(distances)


    # Turbine_As_power = np.asarray(Turbine_As_power).reshape(-1,1)
    # Turbine_Bs_power = np.asarray(Turbine_Bs_power).reshape(-1,1)

    # ransac = linear_model.RANSACRegressor()
    # ransac.fit(Turbine_As_power, Turbine_Bs_power)

    # general_ransac_x = np.arange(Turbine_As_power.min(), Turbine_As_power.max())[:, np.newaxis]
    # general_ransac_y = ransac.predict(general_ransac_x)


    # jensen_x_axis = []
    # jensen_y_axis = []
    # for i in range(0,len(jensen_line[0])):
    #     power = return_power_value(jensen_line[0][i],powercurve_windspeed_new,power_smooth)
    #     jensen_x_axis.append(power)

    #     power = return_power_value(jensen_line[1][i],powercurve_windspeed_new,power_smooth)
    #     jensen_y_axis.append(power)

    # plt.figure(1000)
    # plt.plot(general_ransac_x,general_ransac_y,color='red', linewidth=2,label='RANSAC regressor')
    # plt.plot(jensen_x_axis,jensen_y_axis,color='green', linewidth=1,label='Jensen Prediction')
    # plt.legend(loc="upper left")
    # plt.xlabel('Turbine A Power (kW)')
    # plt.ylabel('Turbine B Power (kW)')
    # plt.title('Jensen-Predicted Power versus RANSAC of Turbine Data ')
    # plt.grid()
    plt.show()




    return 


def get_graphs_Turbine_B_Speed_Power():

    powercurve_windspeed= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5])
    powercurve_windspeed_new = np.linspace(powercurve_windspeed.min(), powercurve_windspeed.max(), 3000)  
    powercurve_power = [7,53,123,208,309,427,567,732,927,1149,1401,1688,2006,2348,2693,3011,3252,3388,3436,3448,3450]

    spl = make_interp_spline(powercurve_windspeed, powercurve_power, k=1)  # type: BSpline
    #powercurve load in
    power_smooth = spl(powercurve_windspeed_new)

    
    turbine_list = csv.reader(open('turbine_list_46.txt', "r"), delimiter=",")
    next(turbine_list)

    Turbine_As_power = []
    Turbine_Bs_power = []

    i = 1
    distances = []

    for row in turbine_list:
        plt.figure(i)

        lead = row[0]
        behind = row[1]

        distance = get_distances(lead,behind)
        distances.append(distance)

        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        plt.scatter(df['jensen_'+behind+'_windspeed'],df['jensen_'+behind+'_power'],marker='x',s=1,color="blue", label='Turbine Actual Data')
        # plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

        plt.title('Turbine B jensen predicted windspeed vs Turbine B Predicted Jensen power ' + lead + ' ' + behind)
        plt.xlabel('Turbine B Predicted Jensen Windspeed (m/s)')
        plt.ylabel('Turbine B Predicted Power (kW)')
        plt.grid()
        plt.grid()

        # plt.scatter(df['power_inferred_'+lead+'_windspeed'],df['power_inferred_'+behind+'_windspeed'])
        # plt.scatter(df['power_inferred_'+lead+'_windspeed'],df['jensen_'+behind+'_windspeed'])


        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = Turbine_A_power.reshape(-1,1)
        Turbine_B_power = Turbine_B_power.reshape(-1,1)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)
        
        ransac = linear_model.RANSACRegressor()
        ransac.fit(Turbine_A_power, Turbine_B_power)

        print('RANSAC estimator coefficient: ' + str(ransac.estimator_.coef_))

        line_X = np.arange(Turbine_A_power.min(), Turbine_A_power.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        plt.plot(line_X,line_y_ransac,color='red', linewidth=2,label='RANSAC regressor')
        plt.legend(loc="upper left")

        
        i = i+1


    turbine_list = csv.reader(open('turbine_list_226.txt', "r"), delimiter=",")
    next(turbine_list)

    i = 50

    for row in turbine_list:
        plt.figure(i)

        lead = row[0]
        behind = row[1]


        distance = get_distances(lead,behind)
        distances.append(distance)

        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        plt.scatter(df['jensen_'+behind+'_windspeed'],df['jensen_'+behind+'_power'],marker='x',s=1,color="blue", label='Turbine Actual Data')
        # plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

        plt.title('Turbine B jensen predicted windspeed vs Turbine B Predicted Jensen power ' + lead + ' ' + behind)
        plt.xlabel('Turbine B Predicted Jensen Windspeed (m/s)')
        plt.ylabel('Turbine B Predicted Power (kW)')
        plt.grid()

        # plt.scatter(df['power_inferred_'+lead+'_windspeed'],df['power_inferred_'+behind+'_windspeed'])
        # plt.scatter(df['power_inferred_'+lead+'_windspeed'],df['jensen_'+behind+'_windspeed'])


        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = Turbine_A_power.reshape(-1,1)
        Turbine_B_power = Turbine_B_power.reshape(-1,1)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)
        
        ransac = linear_model.RANSACRegressor()
        ransac.fit(Turbine_A_power, Turbine_B_power)

        print('RANSAC estimator coefficient: ' + str(ransac.estimator_.coef_))

        line_X = np.arange(Turbine_A_power.min(), Turbine_A_power.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        plt.plot(line_X,line_y_ransac,color='red', linewidth=2,label='RANSAC regressor')
        plt.legend(loc="upper left")

     
        i = i+1


    turbine_list = csv.reader(open('turbine_list_106.txt', "r"), delimiter=",")
    next(turbine_list)

    i = 100

    for row in turbine_list:
        plt.figure(i)

        lead = row[0]
        behind = row[1]


        distance = get_distances(lead,behind)
        distances.append(distance)


        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        plt.scatter(df['jensen_'+behind+'_windspeed'],df['jensen_'+behind+'_power'],marker='x',s=1,color="blue", label='Turbine Actual Data')
        # plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

        plt.title('Turbine B jensen predicted windspeed vs Turbine B Predicted Jensen power ' + lead + ' ' + behind)
        plt.xlabel('Turbine B Predicted Jensen Windspeed (m/s)')
        plt.ylabel('Turbine B Predicted Power (kW)')
        plt.grid()

        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = Turbine_A_power.reshape(-1,1)
        Turbine_B_power = Turbine_B_power.reshape(-1,1)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)
        
        ransac = linear_model.RANSACRegressor()
        ransac.fit(Turbine_A_power, Turbine_B_power)

        print('RANSAC estimator coefficient: ' + str(ransac.estimator_.coef_))

        line_X = np.arange(Turbine_A_power.min(), Turbine_A_power.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        plt.plot(line_X,line_y_ransac,color='red', linewidth=2,label='RANSAC regressor')
        plt.legend(loc="upper left")

        
        i = i+1




def get_all_data_and_polyfit():

    powercurve_windspeed= np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5])
    powercurve_windspeed_new = np.linspace(powercurve_windspeed.min(), powercurve_windspeed.max(), 5000)  
    powercurve_power = [7,53,123,208,309,427,567,732,927,1149,1401,1688,2006,2348,2693,3011,3252,3388,3436,3448,3450]

    spl = make_interp_spline(powercurve_windspeed, powercurve_power, k=1)  # type: BSpline
    #powercurve load in
    power_smooth = spl(powercurve_windspeed_new)

    


    
    turbine_list = csv.reader(open('turbine_list_46.txt', "r"), delimiter=",")
    next(turbine_list)

    Turbine_As_power = []
    Turbine_Bs_power = []

    Turbine_As_speed = []
    Turbine_Bs_speed = []

    Turbine_As_inferred_speed = []
    Turbine_Bs_inferred_speed = []
    Turbine_Bs_jensen_prediction = []
    


    distances = []
    for row in turbine_list:
        
        lead = row[0]
        behind = row[1]

        distance = get_distances(lead,behind)
        distances.append(distance)

        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')

        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = np.array(Turbine_A_power)
        Turbine_B_power = np.array(Turbine_B_power)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)

        Turbine_A_inferred_speed = np.asarray(df['power_inferred_'+lead+'_windspeed'])
        Turbine_B_inferred_speed = np.asarray(df['power_inferred_'+behind+'_windspeed'])

        Turbine_As_inferred_speed.extend(Turbine_A_inferred_speed)
        Turbine_Bs_inferred_speed.extend(Turbine_B_inferred_speed)

        Turbine_A_speed = np.asarray(df[lead+'_Amb_WindSpeed_Avg'])
        Turbine_B_speed = np.asarray(df[behind+'_Amb_WindSpeed_Avg'])

        Turbine_As_speed.extend(Turbine_A_speed)
        Turbine_Bs_speed.extend(Turbine_B_speed)

        Turbine_Bs_jensen_prediction.extend(np.asarray(df['jensen_'+behind+'_windspeed']))


        


        # coefficients = np.polyfit(Turbine_A_power, Turbine_B_power, 4)
        # polynomial = np.poly1d(coefficients)
        # poly_x = np.linspace(Turbine_A_power.min(),Turbine_A_power.max(),1000)    
        # ys = polynomial(poly_x)
        # print(polynomial)
        # plt.plot(poly_x,ys)
        # plt.plot(poly_x,ys,color='Black', linewidth=3,label='8')
        # plt.scatter(Turbine_A_power,Turbine_B_power,s=1)
        # plt.title(lead + ' ' + behind)
        # plt.grid()
        # i = i + 1

    turbine_list = csv.reader(open('turbine_list_226.txt', "r"), delimiter=",")
    next(turbine_list)

    for row in turbine_list:

        
        lead = row[0]
        behind = row[1]

        distance = get_distances(lead,behind)
        distances.append(distance)

        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')

        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = np.array(Turbine_A_power)
        Turbine_B_power = np.array(Turbine_B_power)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)


        Turbine_A_speed = np.asarray(df[lead+'_Amb_WindSpeed_Avg'])
        Turbine_B_speed = np.asarray(df[behind+'_Amb_WindSpeed_Avg'])

        Turbine_As_speed.extend(Turbine_A_speed)
        Turbine_Bs_speed.extend(Turbine_B_speed)

        Turbine_A_inferred_speed = np.asarray(df['power_inferred_'+lead+'_windspeed'])
        Turbine_B_inferred_speed = np.asarray(df['power_inferred_'+behind+'_windspeed'])

        Turbine_As_inferred_speed.extend(Turbine_A_inferred_speed)
        Turbine_Bs_inferred_speed.extend(Turbine_B_inferred_speed)

        Turbine_Bs_jensen_prediction.extend(np.asarray(df['jensen_'+behind+'_windspeed']))


        
    


        # coefficients = np.polyfit(Turbine_A_power, Turbine_B_power, 4)
        # polynomial = np.poly1d(coefficients)
        # poly_x = np.linspace(Turbine_A_power.min(),Turbine_A_power.max(),1000)    
        # ys = polynomial(poly_x)
        # print(polynomial)
        # plt.plot(poly_x,ys)
        # plt.plot(poly_x,ys,color='Black', linewidth=3,label='8')
        # plt.scatter(Turbine_A_power,Turbine_B_power,s=1)
        # plt.title(lead + ' ' + behind)
        # plt.grid()
        # i = i + 1


    turbine_list = csv.reader(open('turbine_list_106.txt', "r"), delimiter=",")
    next(turbine_list)

    for row in turbine_list:

        
        lead = row[0]
        behind = row[1]

        distance = get_distances(lead,behind)
        distances.append(distance)

        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')

        Turbine_A_power = np.asarray(df[lead+'_Grd_Prod_Pwr_Avg'])
        Turbine_B_power = np.asarray(df[behind+'_Grd_Prod_Pwr_Avg'])
        Turbine_A_power = np.array(Turbine_A_power)
        Turbine_B_power = np.array(Turbine_B_power)

        Turbine_As_power.extend(Turbine_A_power)
        Turbine_Bs_power.extend(Turbine_B_power)

        Turbine_A_speed = np.asarray(df[lead+'_Amb_WindSpeed_Avg'])
        Turbine_B_speed = np.asarray(df[behind+'_Amb_WindSpeed_Avg'])

        Turbine_As_speed.extend(Turbine_A_speed)
        Turbine_Bs_speed.extend(Turbine_B_speed)

        Turbine_A_inferred_speed = np.asarray(df['power_inferred_'+lead+'_windspeed'])
        Turbine_B_inferred_speed = np.asarray(df['power_inferred_'+behind+'_windspeed'])

        Turbine_As_inferred_speed.extend(Turbine_A_inferred_speed)
        Turbine_Bs_inferred_speed.extend(Turbine_B_inferred_speed)

        Turbine_Bs_jensen_prediction.extend(np.asarray(df['jensen_'+behind+'_windspeed']))



        # coefficients = np.polyfit(Turbine_A_power, Turbine_B_power, 4)
        # polynomial = np.poly1d(coefficients)
        # poly_x = np.linspace(Turbine_A_power.min(),Turbine_A_power.max(),1000)    
        # ys = polynomial(poly_x)
        # print(polynomial)
        # plt.plot(poly_x,ys)
        # plt.plot(poly_x,ys,color='Black', linewidth=3,label='8')
        # plt.scatter(Turbine_A_power,Turbine_B_power,s=1)
        # plt.title(lead + ' ' + behind)
        # plt.grid()
        # i = i + 1


    jensen_line = get_smoothed_jensen_line(distances)
    jensen_line_enhanced = get_smoothed_jensen_line_Enhanced_Ct(distances)

    Turbine_As_power = np.asarray(Turbine_As_power)
    Turbine_Bs_power = np.asarray(Turbine_Bs_power)
    Turbine_As_speed = np.asarray(Turbine_As_speed)
    Turbine_Bs_speed = np.asarray(Turbine_Bs_speed)
    


    # ransac = linear_model.RANSACRegressor()
    # ransac.fit(Turbine_As_power, Turbine_Bs_power)

    # general_ransac_x = np.arange(Turbine_As_power.min(), Turbine_As_power.max())[:, np.newaxis]
    # general_ransac_y = ransac.predict(general_ransac_x)

    # print(np.array(Turbine_As_power).shape)

    from numpy.polynomial import Polynomial

    # z = np.poly1d(np.polyfit(Turbine_As_power,Turbine_Bs_power,2))
    # print(z[0])
    # print(z[1])
    # print(z[2])
    # print(z)

    # poly_x = np.linspace(Turbine_As_power.min(),Turbine_As_power.max(),1000)
    
    # ys = polynomial(poly_x)

    # for i in range(0,len(poly_x)):
    #     poly_x_value = poly_x[i]
    #     poly_y_value = z[2]*poly_x_value**2 + z[1]

#4 is good
# 5 looks even better
#6 order polynomial




    coefficients = np.polyfit(Turbine_As_power, Turbine_Bs_power, 6)

    sum_of_square_error = np.sum((np.polyval(np.polyfit(Turbine_As_power, Turbine_Bs_power, 6), Turbine_As_power) - Turbine_Bs_power)**2)
    print(np.sqrt(sum_of_square_error/len(Turbine_As_power)))
    print('^Root Mean Square Error')
    polynomial = np.poly1d(coefficients)
    poly_x = np.linspace(Turbine_As_power.min(),Turbine_As_power.max(),len(Turbine_Bs_power))    
    ys = polynomial(poly_x)
   
    print('polyfitting Curve')
    print('Jensen Start')
    jensen_x_axis = []
    jensen_y_axis = []
    for i in range(0,len(jensen_line[0])):
        power = return_power_value(jensen_line[0][i],powercurve_windspeed_new,power_smooth)
        jensen_x_axis.append(power)
        power = return_power_value(jensen_line[1][i],powercurve_windspeed_new,power_smooth)
        jensen_y_axis.append(power)
        if i % 10 == 0:
            print(i)


    jensen_x_axis_enhanced = []
    jensen_y_axis_enhanced = []
    for i in range(0,len(jensen_line_enhanced[0])):
        power = return_power_value(jensen_line_enhanced[0][i],powercurve_windspeed_new,power_smooth)
        jensen_x_axis_enhanced.append(power)
        power = return_power_value(jensen_line_enhanced[1][i],powercurve_windspeed_new,power_smooth)
        jensen_y_axis_enhanced.append(power)
        if i % 10 == 0:
            print(i)
        
    plt.figure(69)

    
    
    plt.scatter(Turbine_As_inferred_speed, Turbine_Bs_inferred_speed,color=[0.257,0.5195,0.953],s=7,marker='P',label="Powercurve-Corrected Windspeed")
    plt.plot(jensen_line[0],jensen_line[1],linewidth=3,color='red',label="Jensen Prediction OEM Ct Values")
    plt.xlabel('Upstream Turbine Wind Speed (m/s) ', fontsize=16)
    plt.xticks(fontsize= 14)
    plt.yticks(fontsize= 14)
    plt.ylabel('Downstream Turbine Wind Speed (m/s) ', fontsize=16)
    plt.title('Jensen Predicted Wind Speed vs Actual Wake Wind Speed', fontsize=16)
    plt.grid()



    # residuals(poly_x,ys,Turbine_As_power,Turbine_Bs_power,'Power')
    residuals(jensen_line[0],jensen_line[1],Turbine_As_inferred_speed,Turbine_Bs_inferred_speed,'Wind Speed')

    
    


    plt.figure(10)
    print('Jensen End')
    plt.scatter(Turbine_As_power,Turbine_Bs_power,color='green', s=0.5,label='Turbine Data')
    plt.plot(jensen_x_axis,jensen_y_axis,color='red', linewidth=3,label='Jensen Prediction OEM Ct')
    # plt.plot(jensen_x_axis_enhanced,jensen_y_axis_enhanced,color='blue', linewidth=3,label='Jensen Prediction Enhanced Ct')
    # plt.plot(poly_x,ys,color='Black', linewidth=3,label='Poly')
    plt.legend(loc="upper left")
    plt.xlabel('Upstream Turbine Power (kW)',fontsize= 14)
    plt.ylabel('Downstream Turbine Power (kW)',fontsize= 14)
    plt.title("Jensen - Predicted Power",fontsize=16)
    plt.grid()
    plt.show()





# normalised_difference = float(downstream_value - corresponding_model_prediction)/standard_deviation
#     residuals.append(float(normalised_difference))
#     print(str(i)+' Calculating Residuals')

# import scipy.stats as stats
# print(np.array(residuals))
# plt.figure(1000)
# stats.probplot(residuals, dist="norm", plot=plt)

# calculate the residuals 





if __name__ =='__main__':
    main()











# plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="blue", label='Turbine Actual Data')
# # plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

# plt.title('Turbine A Actual Power Versus Turbine B Predicted Power ' + lead + ' ' + behind)
# plt.xlabel('Turbine A Power (kW)')
# plt.ylabel('Turbine B Predicted Power (kW)')





# plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df[behind+'_Grd_Prod_Pwr_Avg'],marker='x',s=1,color="blue", label='Turbine Actual Data')
# # plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

# plt.title('Turbine A Power Versus Inferred Wind Speed ' + lead + ' ' + behind)
# plt.xlabel('Upstream Turbine Power (kW)')
# plt.ylabel('Inferred Wind Speed (m/s)')




# plt.scatter(df['jensen_'+behind+'_windspeed'],df['jensen_'+behind+'_power'],marker='x',s=1,color="blue", label='Turbine Actual Data')
# # plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

# plt.title('Turbine B jensen predicted windspeed vs Turbine B Predicted Jensen power ' + lead + ' ' + behind)
# plt.xlabel('Turbine B Predicted Jensen Windspeed (m/s)')
# plt.ylabel('Turbine B Predicted Power (kW)')
# plt.grid()