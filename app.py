from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load the saved model
model = joblib.load('crop_recommendation_model.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Home route (health check)
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Crop Recommendation API is running.',
        'usage': 'Send a POST request to /predict with N, P, K, temperature, humidity, ph, and rainfall.'
    })

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Required keys
    required_keys = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    missing_keys = [key for key in required_keys if key not in data]

    # Check for missing fields
    if missing_keys:
        return jsonify({'error': f'Missing keys in request: {", ".join(missing_keys)}'}), 400

    try:
        # Extract and convert inputs to appropriate types
        input_features = np.array([[ 
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]])

        # Make prediction
        prediction = model.predict(input_features)

        return jsonify({'recommended_crop': prediction[0]})

    except ValueError:
        return jsonify({'error': 'Invalid input types. Please ensure all values are numeric.'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)


# Made By Sachin Vardhan