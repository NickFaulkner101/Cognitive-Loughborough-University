import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn import linear_model, datasets



#plot same as earlier, for each turbine ararngement

def main():
    get_graphs()


def get_general_ransac():
    return


def get_ransac():

    turbine_list = csv.reader(open('turbine_list_106.txt', "r"), delimiter=",")
    next(turbine_list)


    for row in turbine_list:

        lead = row[0]
        behind = row[1]
        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df[behind+'_Grd_Prod_Pwr_Avg'],marker='x',s=1,color="blue", label='Turbine Actual Data')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')


        ransac = linear_model.RANSACRegressor()
        ransac.fit(df[lead+'_Grd_Prod_Pwr_Avg'], df[behind+'_Grd_Prod_Pwr_Avg'])

        general_ransac_x = np.arange(df[lead+'_Grd_Prod_Pwr_Avg'].min(), df[lead+'_Grd_Prod_Pwr_Avg'].max())[:, np.newaxis]
        general_ransac_y = ransac.predict(general_ransac_x)

        plt.plot(general_ransac_x,general_ransac_y,color='red', linewidth=2)




        # plt.scatter(df['power_inferred_'+lead+'_windspeed'],df['power_inferred_'+behind+'_windspeed'])
        # plt.scatter(df['power_inferred_'+lead+'_windspeed'],df['jensen_'+behind+'_windspeed'])



       




def get_graphs():

    
    turbine_list = csv.reader(open('turbine_list_46.txt', "r"), delimiter=",")
    next(turbine_list)

    Turbine_As_power = []
    Turbine_Bs_power = []

    i = 1

    for row in turbine_list:
        plt.figure(i)

        lead = row[0]
        behind = row[1]
        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df[behind+'_Grd_Prod_Pwr_Avg'],marker='x',s=1,color="blue", label='Turbine Actual Data')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

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

        plt.title('Power Generation vs Predicted Power Generation: ' + lead + ' ' + behind )
        plt.xlabel('Upstream Turbine Power Generation')
        plt.ylabel('Downstream Turbine Power Generation')
        plt.grid()
        i = i+1


    turbine_list = csv.reader(open('turbine_list_226.txt', "r"), delimiter=",")
    next(turbine_list)

    i = 50

    for row in turbine_list:
        plt.figure(i)

        lead = row[0]
        behind = row[1]
        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df[behind+'_Grd_Prod_Pwr_Avg'],marker='x',s=1,color="blue", label='Turbine Actual Data')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

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

        plt.title('Power Generation vs Predicted Power Generation: ' + lead + ' ' + behind )
        plt.xlabel('Upstream Turbine Power Generation')
        plt.ylabel('Downstream Turbine Power Generation')
        plt.grid()
        i = i+1


    turbine_list = csv.reader(open('turbine_list_106.txt', "r"), delimiter=",")
    next(turbine_list)

    i = 100

    for row in turbine_list:
        plt.figure(i)

        lead = row[0]
        behind = row[1]
        df = pd.read_csv('Dataframe_'+lead+'_'+behind+'.csv')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df[behind+'_Grd_Prod_Pwr_Avg'],marker='x',s=1,color="blue", label='Turbine Actual Data')
        plt.scatter(df[lead+'_Grd_Prod_Pwr_Avg'],df['jensen_'+behind+'_power'],marker='x',s=1,color="green", label='Jensen Prediction')

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

        plt.title('Power Generation vs Predicted Power Generation: ' + lead + ' ' + behind )
        plt.xlabel('Upstream Turbine Power Generation')
        plt.ylabel('Downstream Turbine Power Generation')
        plt.grid()
        i = i+1


    Turbine_As_power = np.asarray(Turbine_As_power).reshape(-1,1)
    Turbine_Bs_power = np.asarray(Turbine_Bs_power).reshape(-1,1)

    ransac = linear_model.RANSACRegressor()
    ransac.fit(Turbine_As_power, Turbine_Bs_power)

    general_ransac_x = np.arange(Turbine_As_power.min(), Turbine_As_power.max())[:, np.newaxis]
    general_ransac_y = ransac.predict(general_ransac_x)

    plt.figure(1000)
    plt.plot(general_ransac_x,general_ransac_y)
    plt.show()




    return 



# calculate the residuals 





if __name__ =='__main__':
    main()
