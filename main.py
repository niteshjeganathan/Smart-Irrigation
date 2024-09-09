import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('data.csv', skiprows=5)
X = dataset.iloc[:, [1, 2, 5, 6, 9, 11]].values

# Calculating Average Temperature (average of the first two columns)
avg_temp = np.round(np.mean(X[:, [0, 1]], axis=1), 1)

# Calculating Average Humidity (average of the next two columns, 5 and 6)
avg_humidity = np.round(np.mean(X[:, [2, 3]], axis=1), 1)

# Now update the array, keeping the averaged temperature and humidity
X = np.column_stack((avg_temp, avg_humidity, X[:, 4:]))

print(X)
