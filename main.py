import numpy as np
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('env_data.csv', skiprows=5)
dates = dataset.iloc[:, 0].values
# T_max, T_min, Humidity_1, Humidity_2, WindSpeed, Sunshine Hours
X = dataset.iloc[:, [1, 2, 5, 6, 9, 11]].values  # Extracting the relevant columns  

# Extraction of Windspeed and Sunshine Hours
wind_speed = X[:, -2]  # Wind speed in m/s
sunshine_hours = X[:, -1]  # Sunshine hours

# Calculating Average Temperature
avg_temp = np.round(np.mean(X[:, [0, 1]], axis=1), 1)
T_max = X[:, 0]
T_min = X[:, 1]

# Calculation of Terms required for Penman-Monteith Equation
def calculate_et0(X, T_max, T_min, avg_temp, wind_speed, sunshine_hours, dates): 
    # Constants
    albedo = 0.23  # Typical value for agricultural land
    sigma = 4.903e-9  # Stefan-Boltzmann constant (MJ/m²/day/K⁴)
    psychrometric_constant = 0.066  # Approx. kPa/°C
    G = 0  # Soil heat flux density (MJ/m²/day), assumed to be zero for daily calculations

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

    # Calculate delta (Δ) using average temperature
    delta = np.round(slope_of_saturation_vapor_pressure(avg_temp), 2)

    # Calculate e_a from average humidity
    def actual_vapor_pressure(H, T):
        es = saturation_vapor_pressure(T)
        return (H / 100) * es

    # Assuming average humidity is provided in the dataset
    avg_humidity = np.round(np.mean(X[:, [2, 3]], axis=1), 1)
    e_a = actual_vapor_pressure(avg_humidity, avg_temp)

    # Calculate net shortwave radiation (R_sn)
    R_sn = net_shortwave_radiation(R_s, albedo)

    # Calculate net longwave radiation (R_l)
    R_l = estimate_longwave_radiation(T_max, T_min, e_a, R_s, R_s0)

    # Calculate net radiation (R_n)
    R_n = calculate_net_radiation(R_sn, R_l)

    # Calculating Target Variable
    # Calculate E_t0 using the Penman-Monteith equation
    def calculate_et0(delta, R_n, G, gamma, T, u, e_a):
        return (0.408 * delta * (R_n - G) + gamma * (900 / (T + 273)) * u * (saturation_vapor_pressure(T) - e_a)) / (delta + gamma * (1 + 0.34 * u))

    # Calculate E_t0
    E_t0 = np.round(calculate_et0(delta, R_n, G, psychrometric_constant, avg_temp, wind_speed, e_a), 2)

    # Combine dates, input data (X), and calculated E_t0 into a single array
    result_data = np.column_stack((dates, X, E_t0))

    # Create a DataFrame
    df = pd.DataFrame(result_data, columns=['date', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours', 'E_t0'])
    
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

    # Extract month number and year
    df['month'] = df['date'].dt.month
    df['date'] = df['date'].dt.day

    df = df[['date', 'month', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours', 'E_t0']]

    # Write the DataFrame to a CSV file
    df.to_csv('et0_output.csv', index=False)

    return result_data

data = calculate_et0(X, T_max, T_min, avg_temp, wind_speed, sunshine_hours, dates)
print(data)