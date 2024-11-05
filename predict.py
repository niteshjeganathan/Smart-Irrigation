import numpy as np
import pickle
import pandas as pd

# Load the best model
with open('best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Crop coefficient (Kc) values for different growth stages
# Replace with actual values from FAO guidelines as needed
crop_coefficients = {
    'wheat': {
        'initial': 0.3,
        'development': 0.7,
        'mid-season': 1.15,
        'late-season': 0.5
    },
    'maize': {
        'initial': 0.3,
        'development': 1.2,
        'mid-season': 1.2,
        'late-season': 0.6
    }
    # Add more crops and their Kc values as needed
}

# Growth stage durations in days (example values; adjust as necessary)
growth_stages = {
    'wheat': {
        'initial': 15,
        'development': 25,
        'mid-season': 50,
        'late-season': 30
    },
    'maize': {
        'initial': 20,
        'development': 35,
        'mid-season': 40,
        'late-season': 30
    }
    # Add more crops and their stage durations as needed
}

# Function to determine the current growth stage and Kc value
def get_crop_stage_and_kc(crop, age_in_days):
    stages = growth_stages[crop]
    cumulative_days = 0
    for stage, duration in stages.items():
        cumulative_days += duration
        if age_in_days <= cumulative_days:
            return stage, crop_coefficients[crop][stage], cumulative_days - age_in_days
    return 'late-season', crop_coefficients[crop]['late-season'], 0

# Function to calculate daily water requirement
def predict_daily_et0(model, month, t_max, t_min, humidity_1, humidity_2, wind_speed, sunshine_hours):
    features = np.array([[month, t_max, t_min, humidity_1, humidity_2, wind_speed, sunshine_hours]])
    return model.predict(features)[0]

# Get user inputs
crop_type = input("Enter crop type (e.g., wheat, maize): ").lower()
if crop_type not in crop_coefficients:
    print("Crop type not recognized. Please add the crop and its coefficients to the script.")
    exit()

month = int(input("Enter month (1-12): "))
t_max = float(input("Enter maximum temperature (°C): "))
t_min = float(input("Enter minimum temperature (°C): "))
humidity_1 = float(input("Enter morning humidity (%): "))
humidity_2 = float(input("Enter afternoon humidity (%): "))
wind_speed = float(input("Enter wind speed (m/s): "))
sunshine_hours = float(input("Enter sunshine hours: "))
age_in_days = int(input("Enter age of the crop in days: "))
crop_lifespan = sum(growth_stages[crop_type].values())

# Predict the reference evapotranspiration (ETo) for the given day
eto = predict_daily_et0(best_model, month, t_max, t_min, humidity_1, humidity_2, wind_speed, sunshine_hours)
print(f"\nReference evapotranspiration (ETo) for today: {eto:.2f} mm")

# Daily planner for the rest of the crop's lifecycle
remaining_days = crop_lifespan - age_in_days
planner_data = {
    "Day": [],
    "Growth Stage": [],
    "Days Left in Stage": [],
    "Kc Value": [],
    "Predicted ETo (mm)": [],
    "Predicted ETc (mm)": []
}

for day in range(remaining_days):
    future_age = age_in_days + day + 1
    stage, kc_value, days_left_in_stage = get_crop_stage_and_kc(crop_type, future_age)
    daily_etc = kc_value * eto

    # Append data for the planner
    planner_data["Day"].append(f"Day {future_age}")
    planner_data["Growth Stage"].append(stage)
    planner_data["Days Left in Stage"].append(days_left_in_stage)
    planner_data["Kc Value"].append(kc_value)
    planner_data["Predicted ETo (mm)"].append(eto)
    planner_data["Predicted ETc (mm)"].append(daily_etc)

# Convert planner data to DataFrame and display
planner_df = pd.DataFrame(planner_data)
print("\nDaily Water Requirement Planner:")
print(planner_df)

# Optional: Save planner data to CSV
planner_df.to_csv(f"{crop_type}_daily_water_planner.csv", index=False)
print(f"\nPlanner data saved as '{crop_type}_daily_water_planner.csv'")