from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('xgboost_aqi_model.pkl', 'rb'))

# Pollution Level Bucket
def get_pollution_level(aqi_value):
    if 0 <= aqi_value <= 50:
        return "Good", "Air quality is satisfactory."
    elif 51 <= aqi_value <= 100:
        return "Satisfactory", "Acceptable; some pollutants may be a concern for sensitive people."
    elif 101 <= aqi_value <= 200:
        return "Moderate", "Might affect sensitive individuals."
    elif 201 <= aqi_value <= 300:
        return "Poor", "Unhealthy for sensitive groups."
    else:
        return "Very Poor", "Health alert; everyone may experience health effects."

# Scaling and log transformation for features and AQI target
def scale_and_transform(features, aqi_value):
    # Scaling the features
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)
    
    # Applying log transformation to the AQI value
    log_aqi = np.log(aqi_value)
    
    return scaled_features, log_aqi

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Assume you are passing the feature values through form fields
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        nox = float(request.form['nox'])
        co = float(request.form['co'])
        so2 = float(request.form['so2'])

        # Create a DataFrame for the features (ensure this is the same as the model input shape)
        features = pd.DataFrame([[pm25, pm10, nox, co, so2]], 
                                columns=['PM2.5', 'PM10', 'NOx', 'CO', 'SO2'])

        # Scale and apply log transformation
        scaled_features, _ = scale_and_transform(features, 0)  # Log transformation is done separately for AQI

        # Make the prediction using the model
        aqi_prediction_log = model.predict(scaled_features)[0]
        
        # Reverse the log transformation to get the actual AQI value
        aqi_prediction = np.exp(aqi_prediction_log)

        # Get pollution level
        level, description = get_pollution_level(aqi_prediction)

        # Render the result in the template
        return render_template('index.html', prediction_text=f"AQI: {aqi_prediction:.2f} - {level}",
                               description=description, aqi=aqi_prediction)

if __name__ == "__main__":
    app.run(debug=True)