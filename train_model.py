import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle  # For saving the model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from penman_monteith import calculate_et0  # Importing the function for target calculation

# Load datasets
processed_data = pd.read_csv('processed_data.csv')
cleaned_augmented_data = pd.read_csv('cleaned_augmented_data.csv')

# Combine the datasets
combined_data = pd.concat([processed_data, cleaned_augmented_data], ignore_index=True)

# Prepare features matrix for Penman-Monteith calculation
features = combined_data[['month', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours']].values

# Calculate the target values (E_t0)
combined_data['E_t0'] = calculate_et0(features)

# Split data into features (X) and target (y)
X = combined_data[['month', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours']]
y = combined_data['E_t0']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define parameter grids for each model
param_grids = {
    'Linear Regression': {},  # Linear regression has no hyperparameters for tuning
    'Random Forest': {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'ANN': {
        'hidden_layer_sizes': [(50, 25), (100, 50), (100, 50, 25)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
}

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=0),
    'Decision Tree': DecisionTreeRegressor(random_state=0),
    'ANN': MLPRegressor(max_iter=1000, random_state=0)
}

# Train, tune, and evaluate each model
results = {}
predictions = {}
best_model = None
best_score = -float('inf')  # Initialize with a very low score
best_model_name = ""

for name, model in models.items():
    if param_grids[name]:  # Only apply GridSearchCV if there are parameters to tune
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
    else:
        # For Linear Regression (no parameters to tune)
        best_estimator = model.fit(X_train, y_train)
    
    # Evaluate the best model
    y_pred = best_estimator.predict(X_test)
    predictions[name] = y_pred
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100  # R-squared as a percentage for accuracy
    results[name] = {'Mean Squared Error': mse, 'Accuracy (R-squared)': accuracy}
    
    # Update the best model based on R-squared accuracy
    if accuracy > best_score:
        best_score = accuracy
        best_model = best_estimator
        best_model_name = name

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)

# Save the best model
model_filename = f"best_model_{best_model_name}.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)
print(f"Best model '{best_model_name}' with accuracy {best_score:.2f}% saved as '{model_filename}'.")

# Visualization of each model's predictions against actual values in the same window
plt.figure(figsize=(14, 10))
for i, (name, y_pred) in enumerate(predictions.items(), start=1):
    plt.subplot(2, 2, i)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Diagonal line
    plt.title(f'{name} Predictions vs Actual')
    plt.xlabel('Actual E_t0')
    plt.ylabel('Predicted E_t0')
    plt.grid(True)

plt.tight_layout()
plt.show()
