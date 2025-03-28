from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and components
model_path = Path('backend/models/dyslexia_classifier.pkl')
predictor = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess data
        X = df.copy()
        
        # Convert categorical columns to numeric
        categorical_cols = ['Gender', 'Nativelang', 'Otherlang']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = (X[col] == 'Yes').astype(int)
        
        # Convert Age to numeric and handle missing values
        if 'Age' in X.columns:
            X['Age'] = pd.to_numeric(X['Age'], errors='coerce')
            X['Age'].fillna(X['Age'].median(), inplace=True)
        
        # Select features
        X_selected = predictor.feature_selector.transform(X)
        
        # Scale features
        X_scaled = predictor.scaler.transform(X_selected)
        
        # Make prediction
        prediction = predictor.model.predict(X_scaled)
        probability = predictor.model.predict_proba(X_scaled)
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),  # Probability of being dyslexic
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 