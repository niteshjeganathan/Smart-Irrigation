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

# Calculation of Terms required for Penman-Monteith Equation
def calculate_et0(X): 
    # Constants
    albedo = 0.23  # Typical value for agricultural land
    sigma = 4.903e-9  # Stefan-Boltzmann constant (MJ/m²/day/K⁴)
    psychrometric_constant = 0.066  # Approx. kPa/°C
    G = 0  # Soil heat flux density (MJ/m²/day), assumed to be zero for daily calculations

    # Extract data from the input matrix X
    months = X[:, 0]
    T_max = X[:, 1]
    T_min = X[:, 2]
    Humidity_1 = X[:, 3]
    Humidity_2 = X[:, 4]
    wind_speed = X[:, 5]
    sunshine_hours = X[:, 6]

    # Calculate Average Temperature
    avg_temp = np.round(np.mean([T_max, T_min], axis=0), 1)

    # Step 1: Calculate clear-sky radiation (R_s0) using temperature
    def clear_sky_radiation(T_max, T_min):
        return 0.75 + 2 * 10**(-5) * (T_max + T_min) * 0.0820 

    # Calculate clear-sky radiation (R_s0)
    R_s0 = clear_sky_radiation(T_max, T_min)

    # Step 2: Solar radiation (R_s) using sunshine hours
    def estimate_solar_radiation(S, S0, R_s0, a=0.25, b=0.50):
        return R_s0 * (a + b * (S / S0))

    # Calculate R_s using sunshine hours
    R_s = estimate_solar_radiation(sunshine_hours, 12, R_s0)  # Assuming 12 hours as maximum possible sunshine hours

    # Step 3: Shortwave radiation (R_sn)
    def net_shortwave_radiation(R_s, albedo):
        return (1 - albedo) * R_s

    # Step 4: Approximate longwave radiation (R_l)
    def estimate_longwave_radiation(T_max, T_min, e_a, R_s, R_s0):
        # Convert temperatures to Kelvin
        T_max_k = T_max + 273.16
        T_min_k = T_min + 273.16
        # Approximate net longwave radiation (R_l)
        R_l = sigma * ((T_max_k**4 + T_min_k**4) / 2) * (0.34 - 0.14 * np.sqrt(e_a)) * (1.35 * (R_s / R_s0) - 0.35)
        return R_l

    # Step 5: Calculate Net Radiation (R_n)
    def calculate_net_radiation(R_sn, R_l):
        return R_sn - R_l

    # Step 6: Calculate delta (Δ): Slope of saturation vapor pressure curve
    def saturation_vapor_pressure(T):
        return 0.6108 * np.exp((17.27 * T) / (T + 237.3))

    def slope_of_saturation_vapor_pressure(T):
        es = saturation_vapor_pressure(T)
        return (4098 * es) / (T + 237.3)**2

    # Step 7: Calculate delta (Δ) using average temperature
    delta = np.round(slope_of_saturation_vapor_pressure(avg_temp), 2)

    # Step 8: Calculate e_a from average humidity
    def actual_vapor_pressure(H, T):
        es = saturation_vapor_pressure(T)
        return (H / 100) * es

    # Assuming average humidity is calculated from Humidity_1 and Humidity_2
    avg_humidity = np.round(np.mean([Humidity_1, Humidity_2], axis=0), 1)
    e_a = actual_vapor_pressure(avg_humidity, avg_temp)

    # Shortwave radiation
    R_sn = net_shortwave_radiation(R_s, albedo)

    # Longwave radiation
    R_l = estimate_longwave_radiation(T_max, T_min, e_a, R_s, R_s0)

    # Net radiation
    R_n = calculate_net_radiation(R_sn, R_l)

    # Step 9: Calculating Target Variable
    # Calculate E_t0 using the Penman-Monteith equation
    def calculate_et0(delta, R_n, G, gamma, T, u, e_a):
        return (0.408 * delta * (R_n - G) + gamma * (900 / (T + 273)) * u * (saturation_vapor_pressure(T) - e_a)) / (delta + gamma * (1 + 0.34 * u))

    # Calculate E_t0 for each row in the data
    E_t0 = np.round(calculate_et0(delta, R_n, G, psychrometric_constant, avg_temp, wind_speed, e_a), 2)

    # Combine input data (X) and calculated E_t0 into a single array, excluding the date column
    result_data = np.column_stack((months, T_max, T_min, Humidity_1, Humidity_2, wind_speed, sunshine_hours, E_t0))

    # Create a DataFrame
    df = pd.DataFrame(result_data, columns=['month', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours', 'E_t0'])
    
    # Write the DataFrame to a CSV file
    df.to_csv('et0_output.csv', index=False)

    return result_data

# Call the function with X (without the date column in the final output)
data = calculate_et0(X) 