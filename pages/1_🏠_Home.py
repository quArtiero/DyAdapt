import streamlit as st

st.set_page_config(
    page_title="DyAdapt - Home",
    page_icon="🏠",
    layout="wide"
)

st.title("DyAdapt: Dyslexia Detection and Adaptive Learning System")

# Research Context Section
st.markdown("""
### Research Overview
This application implements a machine learning-based system for early dyslexia detection and adaptive learning support. 
The system combines traditional assessment methods with modern data-driven approaches to provide personalized learning experiences.

### Key Components
1. **Dyslexia Detection**
   - Uses ensemble model (SVM + Random Forest)
   - Implements SMOTE for handling class imbalance
   - Provides confidence scores for predictions
   - Achieves 89.4% accuracy on test set

2. **Adaptive Learning System**
   - Dynamic difficulty adjustment based on performance
   - Real-time feedback and recommendations
   - Progress tracking across multiple sessions

3. **Performance Metrics**
   - Accuracy and response time tracking
   - Intervention level adaptation
   - Learning recommendations generation

### Methodology
- **Data Collection**: Standardized assessment tasks
- **Feature Selection**: Combined ANOVA F-test and Mutual Information
- **Model Training**: Ensemble approach with balanced class weights
- **Adaptive Logic**: Performance-based difficulty adjustment

### Research Questions
1. How accurately can we detect dyslexia using behavioral data?
2. What features are most predictive of dyslexia?
3. How does adaptive learning affect student performance?
4. What patterns emerge in learning progression?

### Research Conclusions
1. **Model Performance**
   - Achieved 89.4% accuracy in dyslexia detection
   - High precision (90%) for non-dyslexic students
   - Conservative approach for dyslexic detection (53% precision)

2. **Key Features**
   - Native language (highest importance)
   - Performance in later sessions (Hits28, Hits25, Score25)
   - Click patterns (Clicks32, Clicks28)

3. **Learning Patterns**
   - Clear progression in accuracy over time
   - Adaptive intervention levels improve performance
   - Personalized learning paths show effectiveness
""")

# Navigation
st.markdown("""
### Navigation
- 🧠 **Dyslexia Prediction**: Make predictions for new students
- 📊 **Student Analysis**: View detailed analysis of student performance
""")

# Disclaimer
st.warning("""
⚠️ **Research Demonstration Only**

This tool is for research demonstration purposes only. It should not be used as the sole basis for clinical diagnosis or educational decisions. 
The system is designed to showcase the potential of machine learning in educational technology and dyslexia research.

For actual dyslexia assessment and intervention, please consult qualified educational and medical professionals.
""") 