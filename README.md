# Cognitive-Loughborough-University
Open the WakeAnalyticalModel folder and download the python file and turbine coordinates .txt file
Run the python script by calling the file followed by the wind direction. 
E.g. "python Jensens_AEP 93" 
Ensure the wind direction is in degrees between 0 to 180 (North through East to South)  or 0 to -180 (North through West to South). 
This model is based on the Jensen's Analytical Model https://www.sciencedirect.com/science/article/abs/pii/S1364032115016123 
The Root Sum Of Squares method is used to calculate wind speeds in cells that are in multiple wind turbine wakes.