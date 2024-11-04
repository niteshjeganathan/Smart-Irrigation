import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

# Load dataset (your processed data)
df_X = pd.read_csv('processed_data.csv')

# Extract features excluding the date column (assuming month is still there)
X = df_X[['month', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours']].values
y = np.zeros(X.shape[0])  # Dummy target (since we only augment data, we don't need an actual target)

# Step 1: Scale only the relevant features
scaler = MinMaxScaler()
X_scaled_features = scaler.fit_transform(X[:, 1:])  # Scale features from index 1 onwards (T_max to Sunshine Hours)

# Combine the scaled features with the month column
X_scaled = np.hstack((X[:, :1], X_scaled_features))  # Keep the month values intact

# Step 2: Define a fitness function using SVR (optional for evaluating the quality of synthetic data)
def svr_fitness(individual, X_train, y_train):
    synthetic_X = np.array(individual).reshape(1, -1)
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    synthetic_pred = svr.predict(synthetic_X)
    expected_value = 0
    fitness = (synthetic_pred - expected_value) ** 2  # Mean Squared Error for synthetic prediction
    return fitness[0],  # Return as a tuple

# Step 3: Define genetic algorithm components

# Define fitness function for minimization (MSE)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Modify individual creation to change the month value by ±1
def create_individual_from_original(row):
    # Change the month value by ±1
    original_month = row[0]
    if original_month == 1:
        perturbed_month = 2  # If it's January, change to February
    elif original_month == 12:
        perturbed_month = 11  # If it's December, change to November
    else:
        perturbed_month = original_month + random.choice([-1, 1])  # Change month normally

    # Keep the perturbed month and use the rest of the original features
    perturbed_values = [
        perturbed_month,  # Perturbed month as integer
        round(row[1] * random.uniform(0.99, 1.01), 1),  # T_max rounded to 1 decimal place
        round(row[2] * random.uniform(0.99, 1.01), 1),  # T_min rounded to 1 decimal place
        round(row[3] * random.uniform(0.99, 1.01), 1),  # Humidity_1 rounded to 1 decimal place
        round(row[4] * random.uniform(0.99, 1.01), 1),  # Humidity_2 rounded to 1 decimal place
        round(row[5] * random.uniform(0.99, 1.01), 1),  # WindSpeed rounded to 1 decimal place
        round(row[6] * random.uniform(0.99, 1.01), 1)   # Sunshine Hours rounded to 1 decimal place
    ]

    return creator.Individual(perturbed_values)

# Population setup (create individuals based on the original dataset)
def create_population_from_original(data, n):
    return [create_individual_from_original(row) for row in random.sample(list(data), n)]

toolbox.register("population", create_population_from_original)

# Evaluation function: SVR
toolbox.register("evaluate", svr_fitness, X_train=X_scaled, y_train=y)

# Genetic operations: Crossover, mutation, and selection
def custom_crossover(ind1, ind2):
    # Crossover for features excluding the month (index 0)
    for i in range(1, len(ind1)):  # Start from index 1 to avoid the month
        if random.random() < 0.5:  # 50% chance to swap
            ind1[i], ind2[i] = ind2[i], ind1[i]  # Swap genes between parents
    return ind1, ind2  # Return the modified individuals

# Register the crossover function
toolbox.register("mate", custom_crossover)

# Modify the mutation function to ensure month values aren't affected and clamp values
def custom_mutate(individual):
    # Only mutate T_max, T_min, Humidity_1, Humidity_2, WindSpeed, Sunshine Hours
    for i in range(1, len(individual)):  # Start from index 1 to avoid the month
        if random.random() < 0.2:  # 20% chance to mutate each gene
            # Apply Gaussian mutation
            mutated_value = individual[i] + random.gauss(0, 0.05)
            # Clamp the values to be within [0, 1]
            mutated_value = max(0, min(mutated_value, 1))
            individual[i] = round(mutated_value, 1)  # Round to 1 decimal place
    return individual,

toolbox.register("mutate", custom_mutate)  # Register the custom mutation function
toolbox.register("select", tools.selTournament, tournsize=3)

# Step 4: Run the genetic algorithm
def run_ga_for_augmentation(original_data, multiplier=2):
    num_new_rows = len(original_data) * (multiplier - 1)
    pop = create_population_from_original(original_data, n=150)  # Start with a population size of 150
    
    hof = tools.HallOfFame(1)  # Keep track of the best solution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    new_data = []
    
    while len(new_data) < num_new_rows:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, 
                                           stats=stats, halloffame=hof, verbose=True)

        # Extract synthetic data points from the population
        for ind in pop:
            synthetic_data_point = np.array(ind).reshape(1, -1)  # Convert individual to array
            new_data.append(synthetic_data_point)

        # Limit the size of the new data to the desired number of rows
        new_data = new_data[:num_new_rows]

    return np.vstack(new_data)

# Step 5: Run GA to augment the data (doubling the dataset size)
synthetic_data_scaled = run_ga_for_augmentation(X_scaled, multiplier=2)  # Double the dataset excluding month
print(f"Generated {len(synthetic_data_scaled)} synthetic data points.")

# Inverse transform the synthetic data to original scale
# Only inverse transform the scaled features
synthetic_data_features = scaler.inverse_transform(synthetic_data_scaled[:, 1:])  # Inverse transform only the features
synthetic_data = np.hstack((synthetic_data_scaled[:, :1], synthetic_data_features))  # Combine month with inverse scaled features

# Combine final data with the perturbed months (already included in synthetic data)
final_df = pd.DataFrame(synthetic_data, columns=['month', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours'])

# Round the final DataFrame values to one decimal place
final_df = final_df.round(1)

# Step 6: Validation to remove out-of-range rows
def validate_and_remove_out_of_range(original_df, augmented_df):
    valid_mask = np.ones(len(augmented_df), dtype=bool)  # Start with all True (valid)
    removed_rows_count = 0  # Counter for removed rows

    for column in augmented_df.columns:
        min_val = original_df[column].min()
        max_val = original_df[column].max()

        # Update the mask where values are out of range
        out_of_range_mask = ~augmented_df[column].between(min_val, max_val)  # Identify out-of-range values
        valid_mask &= ~out_of_range_mask  # Keep track of valid rows
        removed_rows_count += out_of_range_mask.sum()  # Count the number of removed rows

    # Create a new DataFrame with only the valid rows
    filtered_augmented_df = augmented_df[valid_mask]

    return filtered_augmented_df, removed_rows_count

# Run the validation and filtering
cleaned_augmented_data, rows_removed = validate_and_remove_out_of_range(df_X, final_df)

# Print the number of rows removed
print(f"Number of rows removed: {rows_removed}")

# Save the cleaned augmented dataset
cleaned_augmented_data.to_csv('cleaned_augmented_data.csv', index=False)
print("Cleaned augmented dataset saved to 'cleaned_augmented_data.csv'.")
