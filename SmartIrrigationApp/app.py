from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# Load the .keras model
model = tf.keras.models.load_model("./assets/model.keras")

# Load scalers used during training (assume these are saved as pickle files)
with open("./assets/scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)

with open("./assets/scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from the POST request
        data = request.json
        if not data or 'inputs' not in data:
            return jsonify({'error': 'Invalid input data'}), 400

        # Prepare the input data for the model
        inputs = np.array(data['inputs'], dtype=np.float32).reshape(1, -1)  # Input as 2D array (1, features)
        
        # Scale the input data using the scaler from training
        scaled_inputs = scaler_X.transform(inputs)
        
        # Reshape for the LSTM model (LSTM expects 3D input: [batch_size, timesteps, features])
        reshaped_inputs = scaled_inputs.reshape(1, 1, -1)

        # Run inference
        scaled_prediction = model.predict(reshaped_inputs)
        
        # Inverse transform the prediction to get the original scale
        prediction = scaler_y.inverse_transform(scaled_prediction)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
