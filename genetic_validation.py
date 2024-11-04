import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load original and cleaned augmented datasets
original_data = pd.read_csv('processed_data.csv')  # Ensure this path is correct
augmented_data = pd.read_csv('cleaned_augmented_data.csv')  # Ensure this path is correct

# Ensure the data is sorted by time if applicable
# Assuming there's a 'date' column; uncomment the next line if necessary
# original_data.sort_values(by='date', inplace=True)
# augmented_data.sort_values(by='date', inplace=True)

# Function to check if augmented values are within the original value ranges
def check_value_ranges(original_df, augmented_df):
    out_of_range_count = 0
    for column in original_df.columns:
        min_val = original_df[column].min()
        max_val = original_df[column].max()
        out_of_range = augmented_df[(augmented_df[column] < min_val) | (augmented_df[column] > max_val)]
        out_of_range_count += len(out_of_range)
        if len(out_of_range) > 0:
            print(f"Column '{column}' has {len(out_of_range)} out-of-range values.")
    print(f"Total out-of-range values across all columns: {out_of_range_count}")

# Function to compare mean and standard deviation
def compare_statistics(original_df, augmented_df):
    for column in original_df.columns:
        orig_mean = original_df[column].mean()
        aug_mean = augmented_df[column].mean()
        orig_std = original_df[column].std()
        aug_std = augmented_df[column].std()
        
        print(f"Feature: {column}")
        print(f"  Original Mean: {orig_mean:.2f}, Augmented Mean: {aug_mean:.2f}")
        print(f"  Original Std: {orig_std:.2f}, Augmented Std: {aug_std:.2f}")
        print(f"  Difference in Means: {abs(orig_mean - aug_mean):.2f}")
        print(f"  Difference in Stds: {abs(orig_std - aug_std):.2f}\n")

# Function to compare rolling statistics
def compare_rolling_statistics(original_df, augmented_df, window=3):
    for column in original_df.columns:
        orig_rolling_mean = original_df[column].rolling(window=window).mean()
        aug_rolling_mean = augmented_df[column].rolling(window=window).mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(orig_rolling_mean, label='Original Rolling Mean', color='blue')
        plt.plot(aug_rolling_mean, label='Augmented Rolling Mean', color='orange')
        plt.title(f'Rolling Mean Comparison for {column} (Window Size: {window})')
        plt.legend()
        plt.show()

# Execute the checks
check_value_ranges(original_data, augmented_data)
compare_statistics(original_data, augmented_data)
compare_rolling_statistics(original_data, augmented_data, window=3)
