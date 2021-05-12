import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from geopy.distance import geodesic
from sklearn import linear_model, datasets
import math
from scipy.interpolate import make_interp_spline, BSpline
import scipy.stats as stats


lead = np.array(pd.read_csv('lead.csv')['lead'])
mid = np.array(pd.read_csv('mid.csv')['mid'])
rear = np.array(pd.read_csv('rear.csv')['rear'])







# Calculate the average
lead_mean_mean = np.mean(lead)
mid_mean = np.mean(mid)
rear_mean = np.mean(rear)


lead_std = np.std(lead)
mid_std = np.std(mid)
rear_std = np.std(rear)


# Create lists for the plot
materials = ['Lead Turbine', 'Middle Turbine', 'Rear Turbine']
x_pos = np.arange(len(materials))
CTEs = [lead_mean_mean, mid_mean, rear_mean]
error = [lead_std, mid_std, rear_std]



fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Coefficient of Thermal Expansion ($\degree C^{-1}$)')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()










