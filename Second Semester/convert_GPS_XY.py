import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utm

#----------------------Converts to Local UTM, which is accurate to a few meters-----------------

df = pd.read_csv("RealTurbineLocations.txt")
for i, row in df.iterrows():
    latitude = row.Latitude
    longitude = row.Longitude

    u = utm.from_latlon(latitude, longitude)

    x = u[0]
    y = u[1] 
    df.at[i,'X'] = x
    df.at[i,'Y'] = y

scaling_x = df['X'].min() - 300 #-300, arbitray number for visuals
scaling_y = df['Y'].min() - 300 #-300, arbitray number for visuals

for i, row in df.iterrows():
    x = row.X
    y = row.Y

    x = x - scaling_x
    y = y - scaling_y 
    df.at[i,'X'] = x
    df.at[i,'Y'] = y

#distance test
turb_A08_x = df.iloc[0].X
turb_A08_y = df.iloc[0].Y

turb_A09_x = df.iloc[1].X
turb_A09_y = df.iloc[1].Y


x_difference = turb_A08_x - turb_A09_x
y_difference = turb_A08_y - turb_A09_y

distance = np.sqrt(x_difference**2 + y_difference**2)
print(distance)

final_df = df[["Turbine", "X", "Y"]]
final_df.to_csv('Rampion.csv', index=False)


plt.figure(1)
plt.plot(df.X,df.Y, 'o', color='blue')
plt.grid()
plt.xlabel('Localised X (m)')
plt.ylabel('Localised Y (m)')
plt.show()


