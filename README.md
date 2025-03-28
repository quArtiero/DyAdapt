# DyAdapt: Dyslexia Detection and Adaptive Learning System

A machine learning-based system for early dyslexia detection and personalized learning support.

## Features

- Early dyslexia detection using ensemble ML models
- Adaptive learning system with personalized interventions
- Performance tracking and analysis
- Multi-page web interface
- Real-time predictions and recommendations

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/quArtiero/DyAdapt.git
cd DyAdapt
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

The application consists of two parts: a backend server and a frontend Streamlit app.

### 1. Start the Backend Server

Open a terminal and run:
```bash
python3 backend/app.py
```
The backend server will start on http://localhost:5001

### 2. Start the Frontend Application

Open another terminal and run:
```bash
streamlit run app.py
```
The Streamlit app will open in your default web browser at http://localhost:8501

## Application Structure

- **Home Page**: Overview and navigation
- **Dyslexia Prediction**: Make predictions for new students
- **Student Analysis**: View detailed analysis of student performance
- **Results**: View model performance and research outcomes

## Usage

1. Navigate through the pages using the sidebar menu
2. Enter student data in the Prediction page
3. View analysis and results in the respective pages
4. Use the download buttons to export data

## Important Notes

- This is a research demonstration tool and should not be used for clinical diagnosis
- Always consult qualified professionals for proper assessment
- The system requires both backend and frontend servers to be running simultaneously

## Troubleshooting

If you encounter any issues:

1. Make sure both servers are running:
   - Backend on port 5001
   - Frontend on port 8501

2. Check for error messages in the terminal

3. Ensure all required packages are installed:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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

### Research Conclusions
1. **Model Performance**
   - Achieved 84.9% accuracy in dyslexia detection
   - Strong performance in identifying non-dyslexic students (95% precision)
   - Moderate performance for dyslexic students (38% precision)
   - Balanced recall rates (88% for non-dyslexic, 60% for dyslexic)

2. **Feature Importance**
   - Identified 50 key predictive features
   - Strong correlation with:
     - Task completion metrics (clicks, hits)
     - Performance scores
     - Native language
     - Response patterns

3. **Adaptive Learning Impact**
   - Dynamic difficulty adjustment improved engagement
   - Personalized recommendations enhanced learning experience
   - Progress tracking enabled data-driven interventions
   - Real-time feedback supported student development

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

## Final Conclusions

This research project demonstrates the potential of machine learning in educational technology, specifically in dyslexia detection and intervention. Key findings include:

1. **Technical Achievement**
   - Successful implementation of ML-based dyslexia detection
   - Effective feature selection and model optimization
   - Robust adaptive learning system

2. **Educational Impact**
   - Personalized learning experiences
   - Data-driven intervention strategies
   - Real-time progress monitoring

3. **Research Value**
   - Insights into behavioral patterns
   - Understanding of learning trajectories
   - Evidence for adaptive intervention effectiveness

4. **Future Directions**
   - Integration with real-world educational settings
   - Expansion of assessment methods
   - Validation with clinical data
   - Development of more sophisticated intervention strategies

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
