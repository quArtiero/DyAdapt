import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="DyAdapt - Dyslexia Intervention Visualization",
    page_icon="🧠",
    layout="wide"
)

# Load data
@st.cache_data
def load_results():
    results_path = Path("backend/output/student_results.json")
    with open(results_path, 'r') as f:
        return json.load(f)

def create_accuracy_plot(sessions):
    """Create accuracy over time plot."""
    df = pd.DataFrame(sessions)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['session_id'],
        y=df['accuracy'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#1f77b4')
    ))
    fig.update_layout(
        title='Accuracy Over Time',
        xaxis_title='Session Number',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1]),
        template='plotly_white'
    )
    return fig

def create_intervention_plot(sessions):
    """Create intervention level over time plot."""
    df = pd.DataFrame(sessions)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['session_id'],
        y=df['intervention_level'],
        mode='lines+markers',
        name='Intervention Level',
        line=dict(color='#ff7f0e')
    ))
    fig.update_layout(
        title='Intervention Level Over Time',
        xaxis_title='Session Number',
        yaxis_title='Level',
        yaxis=dict(range=[1, 5]),
        template='plotly_white'
    )
    return fig

def create_combined_plot(sessions):
    """Create combined accuracy and intervention level plot."""
    df = pd.DataFrame(sessions)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df['session_id'],
            y=df['accuracy'],
            name='Accuracy',
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['session_id'],
            y=df['intervention_level'],
            name='Intervention Level',
            line=dict(color='red')
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Learning Progress Analysis',
        xaxis_title='Session Number',
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="Accuracy", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="Intervention Level", secondary_y=True, range=[1, 5])
    
    return fig

def main():
    st.title("🧠 DyAdapt: Dyslexia Intervention Visualization")
    
    # Research Context
    st.markdown("""
    ### Research Context
    This application demonstrates how machine learning can be used to:
    1. Detect potential dyslexia from student performance data
    2. Simulate learning sessions with adaptive interventions
    3. Track progress and adjust difficulty levels in real-time
    
    **Research Question**: How can machine learning be used to both detect dyslexia and generate personalized, adaptive learning interventions?
    
    **Methodology**:
    - ML model trained on real dyslexia screening data
    - Simulated learning sessions with performance tracking
    - Adaptive intervention system based on student progress
    
    **Disclaimer**: This is a research demonstration tool and should not be used for clinical diagnosis.
    """)
    
    # Load results
    results = load_results()
    
    # Sidebar
    st.sidebar.header("Student Selection")
    student_ids = [f"Student {r['student_id']}" for r in results]
    selected_student = st.sidebar.selectbox(
        "Select a student",
        student_ids
    )
    
    # Get selected student data
    student_idx = int(selected_student.split()[-1]) - 1
    student_data = results[student_idx]
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dyslexia Prediction")
        prediction = "Likely Dyslexic" if student_data['predicted_dyslexia'] == 1 else "Not Likely Dyslexic"
        confidence = student_data['confidence'] * 100
        st.metric(
            "Prediction",
            prediction,
            f"{confidence:.1f}% confidence"
        )
    
    with col2:
        st.subheader("Learning Progress")
        st.plotly_chart(create_accuracy_plot(student_data['sessions']), use_container_width=True)
    
    st.subheader("Intervention Adaptation")
    st.plotly_chart(create_intervention_plot(student_data['sessions']), use_container_width=True)
    
    # Combined visualization
    st.subheader("Combined Progress View")
    st.plotly_chart(create_combined_plot(student_data['sessions']), use_container_width=True)
    
    # Session details
    st.subheader("Session Details")
    sessions_df = pd.DataFrame(student_data['sessions'])
    st.dataframe(sessions_df)
    
    # Model Information
    st.markdown("---")
    st.markdown("""
    ### Model Information
    The dyslexia detection model uses:
    - Support Vector Machine (SVM) classifier
    - Feature selection to identify key predictors
    - SMOTE for handling class imbalance
    - Standardized feature scaling
    
    The intervention system:
    - Adapts difficulty based on performance
    - Considers both accuracy and response time
    - Provides detailed feedback for each session
    """)

if __name__ == "__main__":
    main() 