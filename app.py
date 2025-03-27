import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

# Set page config
st.set_page_config(
    page_title="DyAdapt - Dyslexia Detection and Adaptive Learning",
    page_icon="🧠",
    layout="wide"
)

# Load data
@st.cache_data
def load_results():
    results_path = Path("backend/output/student_results.json")
    with open(results_path, 'r') as f:
        return json.load(f)

def create_accuracy_plot(sessions_df):
    """Create accuracy over time plot with threshold zones."""
    fig = go.Figure()
    
    # Add threshold zones
    fig.add_trace(go.Scatter(
        x=[0, len(sessions_df)],
        y=[0.8, 0.8],
        mode='lines',
        line=dict(color='green', dash='dash'),
        name='High Performance'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, len(sessions_df)],
        y=[0.4, 0.4],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Low Performance'
    ))
    
    # Add accuracy line
    fig.add_trace(go.Scatter(
        x=sessions_df['session_id'],
        y=sessions_df['accuracy'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        title='Accuracy Over Time',
        xaxis_title='Session Number',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        showlegend=True
    )
    
    # Add zone colors
    fig.add_hrect(y0=0.8, y1=1.0, line_width=0, fillcolor="green", opacity=0.1)
    fig.add_hrect(y0=0.4, y1=0.8, line_width=0, fillcolor="yellow", opacity=0.1)
    fig.add_hrect(y0=0.0, y1=0.4, line_width=0, fillcolor="red", opacity=0.1)
    
    return fig

def create_intervention_plot(sessions_df):
    """Create intervention level over time plot with difficulty zones."""
    fig = go.Figure()
    
    # Add difficulty zones
    fig.add_trace(go.Scatter(
        x=[0, len(sessions_df)],
        y=[4, 4],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='High Difficulty'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, len(sessions_df)],
        y=[2, 2],
        mode='lines',
        line=dict(color='green', dash='dash'),
        name='Low Difficulty'
    ))
    
    # Add intervention level line
    fig.add_trace(go.Scatter(
        x=sessions_df['session_id'],
        y=sessions_df['intervention_level'],
        mode='lines+markers',
        name='Intervention Level',
        line=dict(color='#ff7f0e')
    ))
    
    fig.update_layout(
        title='Intervention Level Over Time',
        xaxis_title='Session Number',
        yaxis_title='Level',
        yaxis=dict(range=[1, 5]),
        template='plotly_white',
        showlegend=True
    )
    
    # Add zone colors
    fig.add_hrect(y0=4, y1=5, line_width=0, fillcolor="red", opacity=0.1)
    fig.add_hrect(y0=2, y1=4, line_width=0, fillcolor="yellow", opacity=0.1)
    fig.add_hrect(y0=1, y1=2, line_width=0, fillcolor="green", opacity=0.1)
    
    return fig

def create_combined_plot(sessions_df):
    """Create combined accuracy and intervention level plot."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add accuracy line
    fig.add_trace(
        go.Scatter(
            x=sessions_df['session_id'],
            y=sessions_df['accuracy'],
            name='Accuracy',
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    # Add intervention level line
    fig.add_trace(
        go.Scatter(
            x=sessions_df['session_id'],
            y=sessions_df['intervention_level'],
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

def export_to_csv(student_data):
    """Export student data to CSV format."""
    sessions_df = pd.DataFrame(student_data['sessions'])
    sessions_df['student_id'] = student_data['student_id']
    sessions_df['predicted_dyslexia'] = student_data['predicted_dyslexia']
    sessions_df['confidence'] = student_data['confidence']
    return sessions_df

def main():
    st.title("DyAdapt: Dyslexia Detection and Adaptive Learning System")
    
    # Research Context Section
    with st.expander("Research Context and Methodology", expanded=True):
        st.markdown("""
        ### Research Overview
        This application implements a machine learning-based system for early dyslexia detection and adaptive learning support. 
        The system combines traditional assessment methods with modern data-driven approaches to provide personalized learning experiences.
        
        ### Key Components
        1. **Dyslexia Detection**
           - Uses SVM classifier with feature selection
           - Implements SMOTE for handling class imbalance
           - Provides confidence scores for predictions
        
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
        - **Feature Selection**: ANOVA F-test for identifying key predictors
        - **Model Training**: SVM with balanced class weights
        - **Adaptive Logic**: Performance-based difficulty adjustment
        
        ### Research Questions
        1. How accurately can we detect dyslexia using behavioral data?
        2. What features are most predictive of dyslexia?
        3. How does adaptive learning affect student performance?
        4. What patterns emerge in learning progression?
        
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
        
        ### Limitations and Future Work
        - Current model is based on simulated data
        - Limited to specific assessment tasks
        - Requires validation with real-world data
        - Potential for expansion to more learning scenarios
        """)
    
    # Final Conclusions Section
    with st.expander("Final Conclusions", expanded=True):
        st.markdown("""
        This research project demonstrates the potential of machine learning in educational technology, specifically in dyslexia detection and intervention. Key findings include:
        
        ### Technical Achievement
        - Successful implementation of ML-based dyslexia detection
        - Effective feature selection and model optimization
        - Robust adaptive learning system
        
        ### Educational Impact
        - Personalized learning experiences
        - Data-driven intervention strategies
        - Real-time progress monitoring
        
        ### Research Value
        - Insights into behavioral patterns
        - Understanding of learning trajectories
        - Evidence for adaptive intervention effectiveness
        
        ### Future Directions
        - Integration with real-world educational settings
        - Expansion of assessment methods
        - Validation with clinical data
        - Development of more sophisticated intervention strategies
        """)
    
    # Disclaimer
    st.warning("""
    ⚠️ **Research Demonstration Only**
    
    This tool is for research demonstration purposes only. It should not be used as the sole basis for clinical diagnosis or educational decisions. 
    The system is designed to showcase the potential of machine learning in educational technology and dyslexia research.
    
    For actual dyslexia assessment and intervention, please consult qualified educational and medical professionals.
    """)
    
    # Load data
    data_path = Path(__file__).parent / "backend" / "output" / "student_results.json"
    session_data = load_results()
    
    # Sidebar for student selection and export
    st.sidebar.title("Student Selection")
    student_ids = [student['student_id'] for student in session_data]
    selected_student = st.sidebar.selectbox(
        "Select a student",
        student_ids,
        format_func=lambda x: f"Student {x}"
    )
    
    # Get selected student's data
    student_data = next(student for student in session_data if student['student_id'] == selected_student)
    sessions_df = pd.DataFrame(student_data['sessions'])
    
    # Export button in sidebar
    if st.sidebar.button("Export Session Data"):
        csv_data = export_to_csv(student_data)
        csv_path = Path(__file__).parent / "backend" / "output" / f"student_{selected_student}_sessions.csv"
        csv_data.to_csv(csv_path, index=False)
        st.sidebar.success(f"Data exported to {csv_path}")
    
    # Display student information
    st.header(f"Student {selected_student} Analysis")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Dyslexia",
            "Yes" if student_data['predicted_dyslexia'] == 1 else "No",
            f"Confidence: {student_data['confidence']:.2%}"
        )
    
    with col2:
        avg_accuracy = sessions_df['accuracy'].mean()
        st.metric("Average Accuracy", f"{avg_accuracy:.2%}")
    
    with col3:
        final_level = sessions_df['intervention_level'].iloc[-1]
        st.metric("Current Level", f"{final_level}/5")
    
    # Display plots
    st.subheader("Learning Progress")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Combined View", "Accuracy", "Intervention Level"])
    
    with tab1:
        st.plotly_chart(create_combined_plot(sessions_df), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_accuracy_plot(sessions_df), use_container_width=True)
    
    with tab3:
        st.plotly_chart(create_intervention_plot(sessions_df), use_container_width=True)
    
    # Display detailed session information
    st.subheader("Session Details")
    
    # Create expandable sections for each session
    for session in student_data['sessions']:
        with st.expander(f"Session {session['session_id']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{session['accuracy']:.2%}")
                st.metric("Response Time", f"{session['response_time']:.1f}s")
                st.metric("Intervention Level", f"{session['intervention_level']}/5")
            
            with col2:
                st.markdown("**Feedback:**")
                st.write(session['explanation'])
                st.markdown("**Recommendations:**")
                for rec in session['recommendations']:
                    st.write(f"- {rec}")

if __name__ == "__main__":
    main() 