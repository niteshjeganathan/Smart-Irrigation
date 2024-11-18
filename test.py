import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix format required by XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    "objective": "multi:softmax",  # Multiclass classification
    "num_class": 3,               # Number of classes in the dataset
    "eval_metric": "mlogloss",    # Multi-class log loss
    "max_depth": 4,               # Depth of the tree
    "eta": 0.1,                   # Learning rate
    "seed": 42
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=50)

# Predict on the test set
y_pred = bst.predict(dtest)
y_pred = y_pred.astype(int)  # Ensure predictions are integers

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
