import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="DyAdapt - Home",
    page_icon="🏠",
    layout="wide"
)

# Title and Introduction
st.title("🏠 Welcome to DyAdapt")
st.markdown("""
### A Machine Learning-Based System for Dyslexia Detection and Adaptive Learning Support

DyAdapt combines traditional assessment methods with data-driven approaches to provide early dyslexia detection and personalized learning support.
""")

# Key Features
st.subheader("Key Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🧠 Dyslexia Detection**
    - Early screening using ML models
    - High accuracy (89.4% on test set)
    - Ensemble approach (SVM + Random Forest)
    - SMOTE for class imbalance
    """)

with col2:
    st.markdown("""
    **📊 Performance Analysis**
    - Detailed student progress tracking
    - Group comparison analytics
    - Intervention effectiveness metrics
    - Learning pattern visualization
    """)

with col3:
    st.markdown("""
    **🎯 Adaptive Learning**
    - Personalized intervention levels
    - Dynamic difficulty adjustment
    - Progress monitoring
    - Performance-based recommendations
    """)

# System Overview
st.subheader("System Overview")
st.markdown("""
### How It Works

1. **Data Collection**
   - Student performance metrics
   - Learning session data
   - Demographic information
   - Progress tracking

2. **Analysis & Detection**
   - Feature selection and preprocessing
   - Model training and validation
   - Performance evaluation
   - Risk assessment

3. **Intervention Support**
   - Adaptive learning paths
   - Personalized exercises
   - Progress monitoring
   - Performance feedback
""")

# Research Context
st.subheader("Research Context")
st.markdown("""
### Key Research Questions

1. **Detection Accuracy**
   - How accurately can we identify dyslexia risk?
   - What features are most predictive?
   - How does the model perform across demographics?

2. **Learning Patterns**
   - What patterns emerge in learning progression?
   - How do intervention strategies affect outcomes?
   - What factors influence improvement rates?

3. **Intervention Effectiveness**
   - Which strategies are most effective?
   - How does personalization impact results?
   - What is the optimal progression path?
""")

# Navigation Guide
st.subheader("Navigation Guide")
st.markdown("""
### Available Pages

1. **🧠 Dyslexia Prediction**
   - Input student data
   - Get risk assessment
   - View confidence scores

2. **📊 Student Analysis**
   - Track individual progress
   - Compare group performance
   - Analyze intervention effectiveness

3. **📈 Results**
   - View model performance
   - Access learning outcomes
   - Download analysis reports
""")

# Disclaimer
st.warning("""
⚠️ **Important Disclaimer**

This tool is for research and demonstration purposes only. It should not be used for clinical diagnosis or educational decisions. Always consult with qualified professionals for proper assessment and support.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed for Research Purposes | Version 1.0</p>
</div>
""", unsafe_allow_html=True) 