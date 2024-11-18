import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle  # For saving the model
import os  # For creating directories
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor  # XGBoost import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import lite
from google.colab import drive  # For Google Drive integration
from penman_monteith import calculate_et0  # Importing the function for target calculation

# Mount Google Drive
drive.mount('/content/drive')
google_drive_path = '/content/drive/MyDrive/models/'  # Path to save models
os.makedirs(google_drive_path, exist_ok=True)  # Create folder if it doesn't exist

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

# Scale the features for all models
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)  # 2D scaled data for traditional models
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Reshape features for LSTM (LSTM expects 3D input: [samples, timesteps, features])
X_scaled_3D = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split data into training and testing sets
X_train_2D, X_test_2D, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=0)
X_train_3D, X_test_3D = train_test_split(X_scaled_3D, test_size=0.2, random_state=0)

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
    },
    'XGBoost': {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
}

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=0),
    'Decision Tree': DecisionTreeRegressor(random_state=0),
    'ANN': MLPRegressor(max_iter=1000, random_state=0),
    'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=0)
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
        grid_search.fit(X_train_2D, y_train)
        best_estimator = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
    else:
        # For Linear Regression (no parameters to tune)
        best_estimator = model.fit(X_train_2D, y_train)
    
    # Evaluate the best model
    y_pred = best_estimator.predict(X_test_2D)
    predictions[name] = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    mse = mean_squared_error(scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(y_pred.reshape(-1, 1)))
    r2 = r2_score(scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(y_pred.reshape(-1, 1)))
    accuracy = r2 * 100  # R-squared as a percentage for accuracy
    results[name] = {'Mean Squared Error': mse, 'Accuracy (R-squared)': accuracy}
    
    # Update the best model based on R-squared accuracy
    if accuracy > best_score:
        best_score = accuracy
        best_model = best_estimator
        best_model_name = name

# Add LSTM Model
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train_3D.shape[1], X_train_3D.shape[2])),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

# Train LSTM model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = lstm_model.fit(X_train_3D, y_train, epochs=50, batch_size=16, validation_data=(X_test_3D, y_test), callbacks=[early_stopping], verbose=1)

# Predict and evaluate LSTM
y_pred_lstm_scaled = lstm_model.predict(X_test_3D)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled)
y_test_rescaled = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_rescaled, y_pred_lstm)
r2 = r2_score(y_test_rescaled, y_pred_lstm)
accuracy = r2 * 100

# Display the results
print(f"LSTM Model - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}, Accuracy: {accuracy:.2f}%")

# Convert and save LSTM model as TFLite
tflite_model_filename = os.path.join(google_drive_path, "best_model_lstm.tflite")
converter = lite.TFLiteConverter.from_keras_model(lstm_model)

# Enable support for TensorFlow operations
converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]

# Disable experimental lowering of tensor list operations
converter._experimental_lower_tensor_list_ops = False

# Enable resource variable support
converter.experimental_enable_resource_variables = True

# Convert the model
tflite_model = converter.convert()

# Save TFLite model
with open(tflite_model_filename, 'wb') as f:
    f.write(tflite_model)

print(f"LSTM model saved as TFLite to Google Drive at '{tflite_model_filename}'.")