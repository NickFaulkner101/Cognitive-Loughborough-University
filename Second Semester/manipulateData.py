import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load in WindSpeed Data
df = pd.read_csv("WindSpeed_Average.csv")
df.index=df['timestamp']
df = df.drop('timestamp', axis =1)
df['WindSpeed_Mean'] = df.mean(axis=1)

#Load in Wind Dir Data
df_Dir = pd.read_csv("WindDir_Data.csv")
df_Dir.index=df_Dir['timestamp']
df_Dir = df_Dir.drop('timestamp', axis =1)

#Merge Dataframes
new_df=df.merge(df_Dir,left_index=True,right_index=True)

#Temporarily basing direction data off of the first wind turbine dir data, will average wind direction at a later point

#print(df.loc[(df['WindSpeed_Mean']>=0) & (df['WindSpeed_Mean']< 5)])

#print(new_df.loc[(new_df['A08_Amb_WindDir_Abs_Avg']>=0) & (new_df['A08_Amb_WindDir_Abs_Avg']< 5)])

zero_five_degrees=new_df.loc[(new_df['A08_Amb_WindDir_Abs_Avg']>=0) & (new_df['A08_Amb_WindDir_Abs_Avg']< 5)][['A08_Amb_WindDir_Abs_Avg', 'WindSpeed_Mean']].copy()

# print(zero_five_degrees)

bins = np.linspace(0,20,21)
wind_speed_distribution = zero_five_degrees['WindSpeed_Mean'].value_counts(bins=bins, sort=False)
print(wind_speed_distribution)



#next steps
#link this to the simulation, i.e. x hours at 1ms, y hours at 2ms and calculate total GWh

# plt.figure(1)
# plt.xlabel('Wind Speed m/s ', fontsize=12)
# plt.xticks(fontsize= 12)
# plt.ylabel('Frequency', fontsize=14)
# plt.title('Wind Speed Distribution')
# plt.grid()
# plt.hist(zero_five_degrees['WindSpeed_Mean'], bins=bins, edgecolor="k")
# plt.xticks(bins)
# plt.show()



# find the index of lines that have angles between 0 and 5, append the wind speeds of the other dataframe to the new dataframe

#Create Pandas dataframe, 2 columns are average wind speed and dir data
#new dataframe satisfies if df['winddir'] is 0<x<5








# for i in range(0,len(df)):
#     letter = group[i]
#     dutch_dataframe = pd.read_csv ('Dutch-Roll_Gp'+letter+'.csv')
#     LattAcc = dutch_dataframe['Lat Acc']
#     dutch_yaw_data.append([dutch_dataframe['Roll Rate'],LattAcc])






#To Do
# End goal: For a given direction, x% of the time is this velocity y% of the time is this velocity
# Number of hours wind blew between each wind speed
#First thing is to therefore get a dataframe for 0<direction<5

#So the inputs of our simulation should include the number of hours in a given speed and direction to produce GWh