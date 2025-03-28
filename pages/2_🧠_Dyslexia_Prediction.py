import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import requests

st.set_page_config(
    page_title="DyAdapt - Prediction",
    page_icon="🧠",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / "backend" / "models" / "dyslexia_classifier.pkl"
    return joblib.load(model_path)

def predict_dyslexia(data, model):
    """Make prediction using the trained model."""
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Preprocess data
    X = df.copy()
    
    # Convert categorical columns to numeric
    categorical_cols = ['Gender', 'Nativelang', 'Otherlang']
    for col in categorical_cols:
        if col in X.columns:
            # Convert to lowercase and map to numeric
            X[col] = X[col].str.lower().map({'yes': 1, 'no': 0, 'male': 1, 'female': 0})
    
    # Convert Age to numeric and handle missing values
    if 'Age' in X.columns:
        X['Age'] = pd.to_numeric(X['Age'], errors='coerce')
        X['Age'].fillna(X['Age'].median(), inplace=True)
    
    # Create a list of all required features in the correct order
    required_features = ['Gender', 'Nativelang', 'Otherlang', 'Age']
    for i in range(1, 33):
        required_features.extend([
            f'Clicks{i}',
            f'Hits{i}',
            f'Misses{i}',
            f'Score{i}',
            f'Accuracy{i}',
            f'Missrate{i}'
        ])
    
    # Create a new DataFrame with all required features in the correct order
    X_ordered = pd.DataFrame()
    
    # Add each feature in the correct order
    for feature in required_features:
        if feature in X.columns:
            X_ordered[feature] = X[feature]
        else:
            X_ordered[feature] = 0
    
    # Scale numeric features
    numeric_features = X_ordered.select_dtypes(include=['float64', 'int64']).columns
    for feature in numeric_features:
        if feature not in ['Gender', 'Nativelang', 'Otherlang']:  # Don't scale categorical features
            # Calculate mean and std for the feature
            mean_val = X_ordered[feature].mean()
            std_val = X_ordered[feature].std()
            
            # Avoid division by zero
            if std_val != 0:
                X_ordered[feature] = (X_ordered[feature] - mean_val) / std_val
            else:
                X_ordered[feature] = 0  # If std is 0, set to 0 instead of NaN
    
    # Fill any remaining NaN values with 0
    X_ordered = X_ordered.fillna(0)
    
    # Make prediction using the pipeline
    prediction = model.predict(X_ordered)
    probability = model.predict_proba(X_ordered)
    
    return int(prediction[0]), float(probability[0][1])

def calculate_session_metrics(accuracy, score):
    """Calculate Clicks, Hits, and Misses based on accuracy and score."""
    # Calculate total clicks based on score (assuming score is out of 100)
    total_clicks = max(1, int(score / 10))  # Ensure at least 1 click
    hits = int(total_clicks * accuracy)
    misses = total_clicks - hits
    return total_clicks, hits, misses

st.title("🧠 Dyslexia Prediction")

# Load model
model = load_model()

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Single Session", "Multiple Sessions"])

with tab1:
    st.subheader("Quick Prediction (Single Session)")
    
    # Basic information
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        nativelang = st.selectbox("Native Language", ["Yes", "No"])
        otherlang = st.selectbox("Other Language", ["Yes", "No"])
        age = st.number_input("Age", min_value=5, max_value=100, value=10)
    
    with col2:
        st.markdown("### Performance Metrics")
        accuracy = st.slider("Accuracy", 0.0, 1.0, 0.7)
        missrate = st.slider("Miss Rate", 0.0, 1.0, 0.2)
        score = st.slider("Score", 0.0, 100.0, 70.0)
    
    if st.button("Make Prediction"):
        # Calculate metrics for first session
        clicks1, hits1, misses1 = calculate_session_metrics(accuracy, score)
        
        # Prepare data
        data = {
            'Gender': gender,
            'Nativelang': nativelang,
            'Otherlang': otherlang,
            'Age': age,
            'Clicks1': clicks1,
            'Hits1': hits1,
            'Misses1': misses1,
            'Score1': score,
            'Accuracy1': accuracy,
            'Missrate1': missrate
        }
        
        # Fill in remaining sessions with same values
        for i in range(2, 33):
            data[f'Clicks{i}'] = clicks1
            data[f'Hits{i}'] = hits1
            data[f'Misses{i}'] = misses1
            data[f'Score{i}'] = score
            data[f'Accuracy{i}'] = accuracy
            data[f'Missrate{i}'] = missrate
        
        # Make prediction
        prediction, probability = predict_dyslexia(data, model)
        
        # Display results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", "Dyslexic" if prediction == 1 else "Non-dyslexic")
        
        with col2:
            st.metric("Confidence", f"{probability:.1%}")
        
        # Add explanation
        st.markdown("""
        ### Interpretation
        - **Prediction**: Based on the input data, the model predicts whether the student is likely to have dyslexia
        - **Confidence**: The probability score indicates how certain the model is about its prediction
        - **Note**: This is a screening tool and should be used in conjunction with professional assessment
        """)

with tab2:
    st.subheader("Detailed Prediction (Multiple Sessions)")
    
    # Basic information
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender (Detailed)", ["Male", "Female"])
        nativelang = st.selectbox("Native Language (Detailed)", ["Yes", "No"])
        otherlang = st.selectbox("Other Language (Detailed)", ["Yes", "No"])
        age = st.number_input("Age (Detailed)", min_value=5, max_value=100, value=10)
    
    with col2:
        st.markdown("### Performance Metrics")
        st.markdown("Enter the values for each session:")
        
        # Create a dictionary to store all session data
        data = {
            'Gender': gender,
            'Nativelang': nativelang,
            'Otherlang': otherlang,
            'Age': age
        }
        
        # Add metrics for all 32 sessions
        for i in range(1, 33):
            st.markdown(f"#### Session {i}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy = st.slider(f"Accuracy", 0.0, 1.0, 0.7, key=f'acc_{i}')
            with col2:
                missrate = st.slider(f"Miss Rate", 0.0, 1.0, 0.2, key=f'miss_{i}')
            with col3:
                score = st.slider(f"Score", 0.0, 100.0, 70.0, key=f'score_{i}')
            
            # Calculate session metrics
            clicks, hits, misses = calculate_session_metrics(accuracy, score)
            
            # Store all metrics
            data[f'Clicks{i}'] = clicks
            data[f'Hits{i}'] = hits
            data[f'Misses{i}'] = misses
            data[f'Score{i}'] = score
            data[f'Accuracy{i}'] = accuracy
            data[f'Missrate{i}'] = missrate
    
    if st.button("Make Detailed Prediction"):
        # Make prediction
        prediction, probability = predict_dyslexia(data, model)
        
        # Display results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", "Dyslexic" if prediction == 1 else "Non-dyslexic")
        
        with col2:
            st.metric("Confidence", f"{probability:.1%}")
        
        # Add explanation
        st.markdown("""
        ### Interpretation
        - **Prediction**: Based on the input data, the model predicts whether the student is likely to have dyslexia
        - **Confidence**: The probability score indicates how certain the model is about its prediction
        - **Note**: This is a screening tool and should be used in conjunction with professional assessment
        """)

# Make prediction request
response = requests.post('http://localhost:5001/predict', json=data) 