import numpy as np
#create a mask to select only the positive readings
readings = np.array([12.5, -3.1, 45.0, 0.0, -8.2, 29.4, 102.1])

#filtering the positive numbers in the array
positive_mask = readings > 0

#making the positive readings
positive_readings = readings[positive_mask]
#printing which numbers in the array are positive and negative
print("Boolean Mask:", positive_mask)
#printing the only positive numbers
print("Positive Readings:", positive_readings)

#readings < 0 finds the negative numbers in readings, then says
#readings[these numbers that are negative] = 0
readings[readings < 0] = 0.0
print("New Readings:", readings)