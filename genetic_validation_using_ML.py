import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from penman_monteith import calculate_et0

# Load training and testing datasets
train_data_path = 'processed_data.csv'
test_data_path = 'cleaned_augmented_data.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Prepare features and calculate E_t0 for training and testing datasets
train_features = train_data[['month', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours']].values
test_features = test_data[['month', 'T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine Hours']].values
train_et0 = calculate_et0(train_features)
test_et0 = calculate_et0(test_features)

# Train a Support Vector Regressor on the training data
svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_model.fit(train_features, train_et0)

# Predict E_t0 values on the test data
svm_predicted_et0 = svm_model.predict(test_features)

# Evaluate the SVM model
svm_mse = mean_squared_error(test_et0, svm_predicted_et0)
svm_r2 = r2_score(test_et0, svm_predicted_et0)
svm_accuracy = svm_r2 * 100  # R-squared as a percentage for accuracy

# Output the results
print("Mean Squared Error (MSE):", svm_mse)
print("Accuracy (R-squared):", svm_accuracy, "%")
