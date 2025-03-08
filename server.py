from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a JSON body with 'emg_data': [list of numeric features]
    Returns a JSON with {'prediction': 0 or 1}
    """
    data = request.get_json()

    # Extract the EMG data from JSON
    emg_data = data.get("emg_data", [])

    # Convert to NumPy array (reshape to 2D if it's a single sample)
    X = np.array(emg_data).reshape(1, -1)

    # Apply the same scaling used during training
    X_scaled = scaler.transform(X)

    # Predict using the loaded SVM model
    prediction = svm_model.predict(X_scaled)

    # Return the prediction as JSON
    return jsonify({"prediction": int(prediction[0])})


if __name__ == "__main__":
    # Run the Flask app (debug=True for development)
    app.run(debug=True)
