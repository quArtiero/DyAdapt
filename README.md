# DyAdapt: Dyslexia Detection and Adaptive Learning System 🧠

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

A machine learning-based system for early dyslexia detection and personalized learning support, combining traditional assessment methods with modern data-driven approaches.

## 🌟 Key Features

- **Early Detection**: Ensemble ML models for dyslexia detection
- **Adaptive Learning**: Personalized interventions and support
- **Real-time Analytics**: Performance tracking and analysis
- **Modern Interface**: Multi-page web application
- **Smart Predictions**: Real-time recommendations and insights

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/quArtiero/DyAdapt.git
cd DyAdapt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the backend server:
```bash
python3 backend/app.py
```
The backend will be available at `http://localhost:5002`

2. Launch the frontend:
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`

## 📱 Application Structure

| Page | Description |
|------|-------------|
| Home | Overview and navigation |
| Dyslexia Prediction | Make predictions for new students |
| Student Analysis | View detailed performance analysis |
| Results | Model performance and research outcomes |

## 💡 Usage Guide

1. Navigate through pages using the sidebar menu
2. Enter student data in the Prediction page
3. View analysis and results in respective pages
4. Export data using download buttons

## 🔧 Technical Architecture

### Core Components

1. **Dyslexia Detection Engine**
   - SVM classifier with feature selection
   - SMOTE for class imbalance handling
   - Confidence scoring system
   - 84.9% overall accuracy

2. **Adaptive Learning System**
   - Dynamic difficulty adjustment
   - Real-time feedback mechanism
   - Progress tracking and analytics
   - Personalized intervention strategies

3. **Performance Analytics**
   - Accuracy tracking
   - Response time analysis
   - Intervention effectiveness metrics
   - Learning trajectory visualization

### Research Methodology

- **Data Collection**: Standardized assessment tasks
- **Feature Selection**: ANOVA F-test
- **Model Training**: SVM with balanced weights
- **Adaptive Logic**: Performance-based adjustment

## 📊 Research Outcomes

### Model Performance
- 84.9% accuracy in dyslexia detection
- 95% precision for non-dyslexic students
- 38% precision for dyslexic students
- 88% recall for non-dyslexic students
- 60% recall for dyslexic students

### Key Features
- 50 identified predictive features
- Strong correlations with:
  - Task completion metrics
  - Performance scores
  - Native language
  - Response patterns

### Adaptive Learning Impact
- Improved student engagement
- Enhanced learning experience
- Data-driven interventions
- Real-time development support

## ⚠️ Important Notes

- This is a research demonstration tool
- Not intended for clinical diagnosis
- Consult qualified professionals for assessment
- Both backend and frontend servers must run simultaneously

## 🔍 Troubleshooting

1. **Server Issues**
   - Verify backend (port 5002) is running
   - Confirm frontend (port 8501) is active
   - Check terminal for error messages

2. **Dependency Problems**
```bash
pip install -r requirements.txt
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Credits

Developed by Pedro Quartiero as part of his AP Capstone Research project.

Dataset source: [Kaggle Dyslexia Dataset](https://www.kaggle.com/datasets/luzrello/dyslexia)

## 🔮 Future Directions

- Integration with educational settings
- Expanded assessment methods
- Clinical data validation
- Enhanced intervention strategies

## ⚠️ Disclaimer

This tool is for research demonstration purposes only. It should not be used as the sole basis for clinical diagnosis or educational decisions.

**Key Limitations:**
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
