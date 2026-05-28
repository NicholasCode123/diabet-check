

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'diabetes_model.joblib'
model_package =  joblib.load(MODEL_PATH)

model = model_package['model']
scaler = model_package['scaler']
medians = model_package['train_medians']
zeros_cols = model_package['cols_with_zeros']
threshold = model_package['threshold']
features = model_package['feature_names']


@app.route('/') 
def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_values = {}
    for feat in features:
        val = data.get(feat)
        input_values[feat] = float(val)


    input_df = pd.DataFrame([input_values])
    for col in zeros_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].replace(0, np.nan)
            input_df[col] = input_df[col].fillna(medians[col])
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = int(prob > threshold)
    if prob < 0.35:
        risk = 'low'
        risk_label = 'Risiko Rendah'
    elif prob < 0.60:
        risk = 'medium'
        risk_label = 'Risiko Sedang'
    else:
        risk = 'high'
        risk_label = 'Risiko Tinggi'
    return jsonify({
        'prediction': prediction,
        'probability': round(float(prob) * 100, 1),
        'risk_level': risk,
        'risk_label': risk_label,
        'diabetic': prediction == 1
    })



if __name__ == '__main__':
    app.run(debug=True, port=5000)
