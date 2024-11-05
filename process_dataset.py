import numpy as np
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('env_data.csv', skiprows=5)

# Extracting month from the date column
dates = pd.to_datetime(dataset.iloc[:, 0], format='%d.%m.%Y')  # Assuming the date format is DD.MM.YYYY
month_column = dates.dt.month  # Extracting month

# T_max, T_min, Humidity_1, Humidity_2, WindSpeed, Sunshine Hours
T_max = dataset.iloc[:, 1].values  # Maximum temperature
T_min = dataset.iloc[:, 2].values  # Minimum temperature
Humidity_1 = dataset.iloc[:, 5].values  # Humidity_1
Humidity_2 = dataset.iloc[:, 6].values  # Humidity_2
WindSpeed = dataset.iloc[:, 9].values  # Wind speed in m/s
Sunshine_Hours = dataset.iloc[:, 11].values  # Sunshine hours

# Combine the extracted columns into a single matrix (without date)
X = np.column_stack((month_column, T_max, T_min, Humidity_1, Humidity_2, WindSpeed, Sunshine_Hours))

# Create a DataFrame to store the data with appropriate column names
df_X = pd.DataFrame(X, columns=['month', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours'])

# Write the DataFrame to a CSV file (without date column)
df_X.to_csv('processed_data.csv', index=False)
print("Processed data saved as 'processed_data.csv'")
