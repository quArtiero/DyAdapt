# DyAdapt: Adaptive ML System for Dyslexia Intervention

DyAdapt is a research-based application that demonstrates how machine learning can be used to both detect dyslexia and generate real-time, feedback-driven learning interventions. It is the applied result of an AP Research project aimed at bridging the gap between diagnosis and personalized support for dyslexic students.

## 🔍 Research Goal
To develop a feedback-driven ML model that detects dyslexia and simulates how adaptive interventions can evolve in response to a student’s progress over time.

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

## 🚨 Disclaimer
This application is a simulation for research purposes only. It is not a diagnostic tool and should not be used as medical advice.

## 🧠 Credits
Developed by Pedro Quartiero as part of his AP Capstone Research project.

Dataset source: https://www.kaggle.com/datasets/luzrello/dyslexia
