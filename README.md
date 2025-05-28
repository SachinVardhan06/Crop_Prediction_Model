# Crop Recommendation System

A machine learning-based web application that recommends suitable crops based on various soil and climate parameters.

## Overview

This system uses machine learning to predict the most suitable crop to grow based on the following parameters:

- N (Nitrogen content in soil)
- P (Phosphorous content in soil)
- K (Potassium content in soil)
- Temperature
- Humidity
- pH (Soil pH)
- Rainfall

## Features

- RESTful API for crop recommendations
- Machine learning model trained on agricultural data
- Easy-to-use interface for predictions
- CORS enabled for cross-origin requests
- Input validation and error handling

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Crop-Prediction
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:

```bash
python app.py
```

2. The API will be available at `http://localhost:5000`

3. Make predictions by sending a POST request to `/predict` with the following JSON structure:

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.879744,
  "humidity": 82.002744,
  "ph": 6.502985,
  "rainfall": 202.935536
}
```

4. The API will respond with the recommended crop:

```json
{
  "recommended_crop": "rice"
}
```

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Prediction endpoint for crop recommendations

## Model Training

The model can be retrained using the `train_model.py` script:

```bash
python train_model.py
```

## Dependencies

- Flask: Web framework
- NumPy: Numerical computing
- Joblib: Model serialization
- Gunicorn: WSGI HTTP Server (for production)

## Contact

- Name: Sachin Vardhan
- LinkedIn: [Harsh Patel](https://www.linkedin.com/in/sachin-vardhan-06/)
