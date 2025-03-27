# DyAdapt: Dyslexia Detection and Adaptive Learning System

A research demonstration tool for dyslexia detection and adaptive learning support, combining machine learning with educational technology.

## Research Context

This application implements a machine learning-based system for early dyslexia detection and adaptive learning support. The system combines traditional assessment methods with modern data-driven approaches to provide personalized learning experiences.

### Key Components
1. **Dyslexia Detection**
   - SVM classifier with feature selection
   - SMOTE for handling class imbalance
   - Confidence scoring system

2. **Adaptive Learning System**
   - Dynamic difficulty adjustment
   - Real-time feedback
   - Progress tracking

3. **Performance Metrics**
   - Accuracy tracking
   - Response time analysis
   - Intervention level adaptation

### Methodology
- Data Collection: Standardized assessment tasks
- Feature Selection: ANOVA F-test
- Model Training: SVM with balanced weights
- Adaptive Logic: Performance-based adjustment

## What This App Shows

The application demonstrates:
1. **Dyslexia Detection**
   - Early identification using behavioral data
   - Confidence-based predictions
   - Feature importance analysis

2. **Learning Progress**
   - Session-by-session tracking
   - Performance visualization
   - Adaptive difficulty adjustment

3. **Student Analytics**
   - Accuracy trends
   - Response time patterns
   - Intervention effectiveness

4. **Research Insights**
   - Performance patterns
   - Learning trajectories
   - Intervention impact

## Disclaimer

This tool is for research demonstration purposes only. It should not be used as the sole basis for clinical diagnosis or educational decisions. The system is designed to showcase the potential of machine learning in educational technology and dyslexia research.

Key limitations:
- Based on simulated data
- Limited assessment scope
- Requires clinical validation
- Not intended for clinical use

For actual dyslexia assessment and intervention, please consult qualified educational and medical professionals.

## 🔍 Research Goal
To develop a feedback-driven ML model that detects dyslexia and simulates how adaptive interventions can evolve in response to a student's progress over time.

## 📁 How it Works

1. **ML Model**:
   - Trained on real data from a gamified dyslexia screening dataset.
   - Predicts whether a student likely has dyslexia.

2. **Session Simulation**:
   - For each student, generates 10 learning sessions.
   - Tracks performance (accuracy, response time) per session.

3. **Adaptive Intervention**:
   - Adjusts intervention difficulty dynamically based on performance trends.
   - Logged and visualized for each student.

4. **Visual Output**:
   - Accuracy and intervention level over time.
   - Individual reports for each student.

## 🛠️ Tech Stack
- Python (pandas, scikit-learn, matplotlib)
- Flask or Streamlit for backend
- HTML/CSS/JS or Streamlit components for frontend

## 🧠 Credits
Developed by Pedro Quartiero as part of his AP Capstone Research project.

Dataset source: https://www.kaggle.com/datasets/luzrello/dyslexia
