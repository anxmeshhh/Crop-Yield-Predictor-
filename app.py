from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load('xgb_crop_yield_model.pkl')
encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        data = {
            'Region': request.form['Region'],
            'Soil_Type': request.form['Soil_Type'],
            'Crop': request.form['Crop'],
            'Weather_Condition': request.form['Weather_Condition'],
            'Fertilizer_Used': int(request.form['Fertilizer_Used']),
            'Irrigation_Used': int(request.form['Irrigation_Used']),
            'Rainfall_mm': float(request.form['Rainfall_mm']),
            'Temperature_Celsius': float(request.form['Temperature_Celsius']),
            'Days_to_Harvest': int(request.form['Days_to_Harvest']),
        }

        # Encode categorical variables
        for col in ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']:
            encoder = encoders[col]
            data[col] = encoder.transform([data[col]])[0]

        # Create DataFrame and match model column order
        input_df = pd.DataFrame([data])
        input_df = input_df[model.get_booster().feature_names]

        # Predict crop yield
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"❌ Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
