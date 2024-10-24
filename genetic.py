import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load dataset (your processed data)
df_X = pd.read_csv('processed_data.csv')

# Extract features including date and month
X = df_X[['date', 'month', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours']].values
y = np.zeros(X.shape[0])  # Dummy target (since we only augment data, we don't need an actual target)

# Check the shape of X
print(f"Original X shape: {X.shape[0]} rows, {X.shape[1]} columns")

# Step 1: Define a fitness function using SVR (optional for evaluating the quality of synthetic data)
def svr_fitness(individual, X_train, y_train):
    # Use the original dataset + the synthetic data (individual)
    synthetic_X = np.array(individual).reshape(1, -1)  # Ensure individual is 2D
    new_X_train = np.vstack((X_train, synthetic_X))  # Add synthetic data to original training set
    
    # Adjust y_train to match the new_X_train size
    new_y_train = np.hstack((y_train, [0]))  # Adding a dummy target for the new synthetic row
    
    # Check if sizes match
    assert new_X_train.shape[0] == new_y_train.shape[0], f"Inconsistent sizes: X_train={new_X_train.shape[0]}, y_train={new_y_train.shape[0]}"
    
    # Train SVR model (optional)
    svr = SVR(kernel='rbf')
    svr.fit(new_X_train, new_y_train)
    
    y_pred = svr.predict(new_X_train)
    
    # Return MSE as the fitness (lower is better)
    mse = mean_squared_error(new_y_train, y_pred)
    
    return mse,

# Step 2: Define genetic algorithm components

# Define fitness function for minimization (MSE)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Manually perturb each feature in the original data for individual creation
def create_individual_from_original(row):
    # Perturb each feature (excluding date and month)
    perturbed_values = [row[2] * random.uniform(0.9, 1.1),  # T_max
                        row[3] * random.uniform(0.9, 1.1),  # T_min
                        row[4] * random.uniform(0.9, 1.1),  # Humidity_1
                        row[5] * random.uniform(0.9, 1.1),  # Humidity_2
                        row[6] * random.uniform(0.9, 1.1),  # WindSpeed
                        row[7] * random.uniform(0.9, 1.1)]  # Sunshine Hours
    return creator.Individual(perturbed_values)

# Population setup (create individuals based on the original dataset)
def create_population_from_original(data, n):
    return [create_individual_from_original(row) for row in random.sample(list(data), n)]

toolbox.register("population", create_population_from_original)

# Evaluation function: SVR
toolbox.register("evaluate", svr_fitness, X_train=X[:, 2:], y_train=y)  # Exclude date and month from model training

# Genetic operations: Crossover, mutation, and selection
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Step 3: Run the genetic algorithm
def run_ga_for_augmentation(original_data, multiplier=2):
    # Calculate the number of new rows required to reach the desired size
    num_new_rows = len(original_data) * (multiplier - 1)
    
    # Create a population based on the original data
    pop = create_population_from_original(original_data, n=50)  # Start with a population size of 50
    
    # Evolutionary algorithm settings
    hof = tools.HallOfFame(1)  # Keep track of the best solution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    # Generating synthetic data
    new_data = []
    
    while len(new_data) < num_new_rows:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, 
                                           stats=stats, halloffame=hof, verbose=True)
        
        # Extract synthetic data points from the population
        for ind in pop:
            synthetic_data_point = np.array(ind).reshape(1, -1)  # Convert individual to array
            new_data.append(synthetic_data_point)
        
        # Limit the size of the new data to the desired number of rows
        new_data = new_data[:num_new_rows]

    return np.vstack(new_data)

# Step 4: Run GA to augment the data (doubling the dataset size)
synthetic_data = run_ga_for_augmentation(X[:, 2:], multiplier=2)  # Double the dataset excluding date and month
print(f"Generated {len(synthetic_data)} synthetic data points.")

# Combine synthetic data with original date and month values
synthetic_dates = np.tile(df_X['date'].values, (synthetic_data.shape[0] // df_X.shape[0]) + 1)[:synthetic_data.shape[0]]
synthetic_months = np.tile(df_X['month'].values, (synthetic_data.shape[0] // df_X.shape[0]) + 1)[:synthetic_data.shape[0]]

# Combine final data with date and month
final_data = np.column_stack((synthetic_dates, synthetic_months, synthetic_data))
final_df = pd.DataFrame(final_data, columns=['date', 'month', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours'])

# Save the augmented dataset
final_df.to_csv('augmented_data_with_dates.csv', index=False)
print("Augmented dataset saved to 'augmented_data_with_dates.csv'.")
