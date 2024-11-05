import numpy as np

def calculate_et0(X):
    # Constants
    albedo = 0.23
    sigma = 4.903e-9
    psychrometric_constant = 0.066
    G = 0

    # Extract data from the input matrix X
    months = X[:, 0]
    T_max = X[:, 1]
    T_min = X[:, 2]
    Humidity_1 = X[:, 3]
    Humidity_2 = X[:, 4]
    wind_speed = X[:, 5]
    sunshine_hours = X[:, 6]

    # Calculate Average Temperature
    avg_temp = np.mean([T_max, T_min], axis=0)

    # Calculate clear-sky radiation (R_s0)
    R_s0 = 0.75 + 2 * 10**(-5) * (T_max + T_min) * 0.0820 

    # Solar radiation (R_s) using sunshine hours
    R_s = R_s0 * (0.25 + 0.50 * (sunshine_hours / 12))

    # Shortwave radiation (R_sn)
    R_sn = (1 - albedo) * R_s

    # Average humidity
    avg_humidity = np.mean([Humidity_1, Humidity_2], axis=0)
    
    # Calculate e_a from average humidity
    e_a = (avg_humidity / 100) * (0.6108 * np.exp((17.27 * avg_temp) / (avg_temp + 237.3)))

    # Longwave radiation (R_l)
    T_max_k = T_max + 273.16
    T_min_k = T_min + 273.16
    R_l = sigma * ((T_max_k**4 + T_min_k**4) / 2) * (0.34 - 0.14 * np.sqrt(e_a)) * (1.35 * (R_s / R_s0) - 0.35)

    # Net radiation (R_n)
    R_n = R_sn - R_l

    # Slope of saturation vapor pressure curve
    es = 0.6108 * np.exp((17.27 * avg_temp) / (avg_temp + 237.3))
    delta = (4098 * es) / (avg_temp + 237.3)**2

    # Calculate E_t0 using the Penman-Monteith equation
    E_t0 = (0.408 * delta * (R_n - G) + psychrometric_constant * (900 / (avg_temp + 273)) * wind_speed * (es - e_a)) / (delta + psychrometric_constant * (1 + 0.34 * wind_speed))

    return np.round(E_t0, 2)
