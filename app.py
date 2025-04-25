from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

app = Flask(__name__)

# Load ML model and encoders
ml_model = joblib.load('xgb_crop_yield_model.pkl')
encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get language preference
        language = request.form.get('language', 'English')

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

        # Create DataFrame and match model input
        input_df = pd.DataFrame([data])
        input_df = input_df[ml_model.get_booster().feature_names]

        # Predict yield
        prediction = round(ml_model.predict(input_df)[0], 2)

        # Prepare input summary for GenAI
        original_inputs = {
            'Region': request.form['Region'],
            'Soil Type': request.form['Soil_Type'],
            'Crop': request.form['Crop'],
            'Weather Condition': request.form['Weather_Condition'],
            'Fertilizer Used (kg/ha)': request.form['Fertilizer_Used'],
            'Irrigation Used (Litre/ha)': request.form['Irrigation_Used'],
            'Rainfall (mm)': request.form['Rainfall_mm'],
            'Temperature (°C)': request.form['Temperature_Celsius'],
            'Days to Harvest': request.form['Days_to_Harvest']
        }

        
        prompt = f"""
        You are an expert agricultural advisor assisting rural farmers.

        Provide a detailed yet simple explanation **strictly in {language} only** based on the following information:
        {original_inputs}
        The predicted crop yield is {prediction} quintals per hectare.

        👉 Format your response using the structure below. All section headers, bullet points, and explanations **must be fully translated into {language}**. Do NOT include any English words, phrases, or symbols.

        🌾 **इनपुट सारांश** (❌ This must be translated to {language})
        - Clearly list each input with a short and simple explanation (✅ Use farmer-friendly terms)

        📊 **पूर्वानुमान उपज** (❌ Translate this too)
        - Mention the predicted yield and explain whether it is low, average, or good in simple terms

        ✅ **क्या अच्छा हुआ** (❌ Translate this heading too)
        - Bullet points of positive or favorable conditions

        ⚠️ **क्या सुधार किया जा सकता है** (❌ Translate this as well)
        - Bullet points listing problems or areas that need improvement

        💡 **विशेषज्ञ की सलाह** (❌ Translate this heading too)
        - 2 to 3 practical and easy-to-understand tips to improve future yield

        🎯 Use only:
        - Fully translated section headers in {language}
        - Simple bullet points (• or -)
        - Friendly emojis
        - Very basic and relatable language for farmers

        ❌ Final response must be written entirely in {language}. Avoid mixing languages. English is **not allowed** anywhere in the response.
        """

        # Generate GenAI response
        gemini_response = model.generate_content(prompt)
        explanation = gemini_response.text

        return render_template('index.html', prediction=prediction, explanation=explanation, language=language)

    except Exception as e:
        return f"❌ Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
