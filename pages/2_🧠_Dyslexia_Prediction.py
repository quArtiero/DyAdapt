import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

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
            X[col] = (X[col] == 'Yes').astype(int)
    
    # Convert Age to numeric and handle missing values
    if 'Age' in X.columns:
        X['Age'] = pd.to_numeric(X['Age'], errors='coerce')
        X['Age'].fillna(X['Age'].median(), inplace=True)
    
    # Make prediction using the pipeline
    prediction = model.predict(X)
    probability = model.predict_proba(X)
    
    return int(prediction[0]), float(probability[0][1])

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
        # Prepare data
        data = {
            'Gender': gender,
            'Nativelang': nativelang,
            'Otherlang': otherlang,
            'Age': age,
            'Accuracy1': accuracy,
            'Missrate1': missrate,
            'Score1': score
        }
        
        # Fill in remaining sessions with same values
        for i in range(2, 33):
            data[f'Accuracy{i}'] = accuracy
            data[f'Missrate{i}'] = missrate
            data[f'Score{i}'] = score
        
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
                data[f'Accuracy{i}'] = st.slider(f"Accuracy", 0.0, 1.0, 0.7, key=f'acc_{i}')
            with col2:
                data[f'Missrate{i}'] = st.slider(f"Miss Rate", 0.0, 1.0, 0.2, key=f'miss_{i}')
            with col3:
                data[f'Score{i}'] = st.slider(f"Score", 0.0, 100.0, 70.0, key=f'score_{i}')
    
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