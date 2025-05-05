import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import requests

st.set_page_config(
    page_title="DyAdapt - Student Analysis",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / "backend" / "data" / "dyslexia_data.csv"
    df = pd.read_csv(data_path, sep=';')
    # Convert Dyslexia column to numeric (Yes=1, No=0)
    df['Dyslexia'] = (df['Dyslexia'] == 'Yes').astype(int)
    return df

st.title("📊 Student Performance Analysis")

# Load data
df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_gender = st.sidebar.multiselect(
    "Select Gender",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

selected_dyslexia = st.sidebar.multiselect(
    "Select Dyslexia Status",
    options=['Dyslexic', 'Non-dyslexic'],
    default=['Dyslexic', 'Non-dyslexic']
)

# Filter data
filtered_df = df[
    (df['Gender'].isin(selected_gender)) &
    (df['Dyslexia'].isin([1 if x == 'Dyslexic' else 0 for x in selected_dyslexia]))
]

# Overview metrics
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Students", len(filtered_df))
with col2:
    st.metric("Dyslexic Students", len(filtered_df[filtered_df['Dyslexia'] == 1]))
with col3:
    st.metric("Non-dyslexic Students", len(filtered_df[filtered_df['Dyslexia'] == 0]))
with col4:
    st.metric("Dyslexia Rate", f"{(len(filtered_df[filtered_df['Dyslexia'] == 1]) / len(filtered_df) * 100):.1f}%")

# Performance Trends
st.subheader("Performance Trends")

# Calculate average metrics by session
avg_metrics = []
for i in range(1, 33):
    avg_metrics.append({
        'Session': i,
        'Accuracy': filtered_df[f'Accuracy{i}'].mean(),
        'Miss Rate': filtered_df[f'Missrate{i}'].mean(),
        'Score': filtered_df[f'Score{i}'].mean()
    })

avg_df = pd.DataFrame(avg_metrics)

# Create performance trend plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=avg_df['Session'],
    y=avg_df['Accuracy'],
    name='Accuracy',
    line=dict(color='#1f77b4')
))
fig.add_trace(go.Scatter(
    x=avg_df['Session'],
    y=avg_df['Score']/100,  # Normalize score to 0-1 range
    name='Score (normalized)',
    line=dict(color='#2ca02c')
))
fig.add_trace(go.Scatter(
    x=avg_df['Session'],
    y=avg_df['Miss Rate'],
    name='Miss Rate',
    line=dict(color='#d62728')
))

fig.update_layout(
    title='Average Performance Metrics Across Sessions',
    xaxis_title='Session',
    yaxis_title='Value',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Performance Distribution
st.subheader("Performance Distribution")

# Create box plots for each metric
col1, col2, col3 = st.columns(3)

with col1:
    fig = px.box(
        filtered_df,
        y=['Accuracy1', 'Accuracy16', 'Accuracy32'],
        title='Accuracy Distribution',
        labels={'value': 'Accuracy', 'variable': 'Session'},
        range_y=[0, 1]  # Accuracy between 0 and 1 (or 0 to 100 if your values are percentages)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        filtered_df,
        y=['Missrate1', 'Missrate16', 'Missrate32'],
        title='Miss Rate Distribution',
        labels={'value': 'Miss Rate', 'variable': 'Session'},
        range_y=[0, 1]  # Again, assume values are proportions. Adjust if you store them as percentages
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = px.box(
        filtered_df,
        y=['Score1', 'Score16', 'Score32'],
        title='Score Distribution',
        labels={'value': 'Score', 'variable': 'Session'},
        range_y=[0, 25]  # Based on your previous score plots, adjust as needed
    )
    st.plotly_chart(fig, use_container_width=True)

# Learning Progress Analysis
st.subheader("Learning Progress Analysis")

# Export Simulation Logs
col1, col2 = st.columns(2)
with col1:
    if st.button("Export Simulation Logs"):
        try:
            API_URL = "http://localhost:5002"
            response = requests.post(f"{API_URL}/export_logs")
            if response.status_code == 200:
                st.success("Simulation logs exported successfully!")
            else:
                st.error("Failed to export logs")
        except Exception as e:
            st.error(f"Error exporting logs: {str(e)}")

with col2:
    # Add download button for the current analysis
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Analysis CSV",
        data=csv_data,
        file_name="student_analysis.csv",
        mime="text/csv"
    )

# Group Comparison
st.subheader("Dyslexic vs Non-dyslexic Learning Patterns")

try:
    API_URL = "http://localhost:5002"
    response = requests.get(f"{API_URL}/compare_groups")
    if response.status_code == 200:
        comparison_data = response.json()
        
        # Create accuracy comparison plot
        fig = go.Figure()
        
        # Add dyslexic group
        fig.add_trace(go.Scatter(
            x=list(range(1, len(comparison_data['dyslexic']['accuracy']) + 1)),
            y=comparison_data['dyslexic']['accuracy'],
            name='Dyslexic Students',
            line=dict(color='#1f77b4')
        ))
        
        # Add non-dyslexic group
        fig.add_trace(go.Scatter(
            x=list(range(1, len(comparison_data['non_dyslexic']['accuracy']) + 1)),
            y=comparison_data['non_dyslexic']['accuracy'],
            name='Non-dyslexic Students',
            line=dict(color='#2ca02c')
        ))
        
        fig.update_layout(
            title='Average Accuracy by Session',
            xaxis_title='Session',
            yaxis_title='Accuracy',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create intervention level comparison plot
        fig = go.Figure()
        
        # Add dyslexic group
        fig.add_trace(go.Scatter(
            x=list(range(1, len(comparison_data['dyslexic']['levels']) + 1)),
            y=comparison_data['dyslexic']['levels'],
            name='Dyslexic Students',
            line=dict(color='#1f77b4')
        ))
        
        # Add non-dyslexic group
        fig.add_trace(go.Scatter(
            x=list(range(1, len(comparison_data['non_dyslexic']['levels']) + 1)),
            y=comparison_data['non_dyslexic']['levels'],
            name='Non-dyslexic Students',
            line=dict(color='#2ca02c')
        ))
        
        fig.update_layout(
            title='Average Intervention Level by Session',
            xaxis_title='Session',
            yaxis_title='Intervention Level',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to fetch group comparison data")
except Exception as e:
    st.error(f"Error fetching group comparison data: {str(e)}")

# Individual Student Learning Curves
st.subheader("Individual Student Learning Curves")

# Select a student to analyze
student_ids = list(filtered_df.index)
selected_student = st.selectbox("Select Student", student_ids)

try:
    API_URL = "http://localhost:5002"
    response = requests.get(f"{API_URL}/learning_curves/{selected_student}")
    if response.status_code == 200:
        curve_data = response.json()
        
        # Create learning curve plot
        fig = go.Figure()
        
        # Add actual intervention levels
        fig.add_trace(go.Scatter(
            x=curve_data['sessions'],
            y=curve_data['actual_levels'],
            name='Actual Progress',
            line=dict(color='#1f77b4')
        ))
        
        # Add ideal curve
        fig.add_trace(go.Scatter(
            x=curve_data['sessions'],
            y=curve_data['ideal_levels'],
            name='Ideal Progress',
            line=dict(color='#2ca02c', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Learning Curve for Student {selected_student}',
            xaxis_title='Session',
            yaxis_title='Intervention Level',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add accuracy progression
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve_data['sessions'],
            y=curve_data['accuracy'],
            name='Accuracy',
            line=dict(color='#d62728')
        ))
        
        fig.update_layout(
            title=f'Accuracy Progression for Student {selected_student}',
            xaxis_title='Session',
            yaxis_title='Accuracy',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to fetch learning curve data")
except Exception as e:
    st.error(f"Error fetching learning curve data: {str(e)}")

# Intervention Types Overview
st.subheader("Intervention Types")
st.markdown("""
### Understanding Intervention Levels
1. **Phonological Support** (Level 1)
   - Highly scaffolded exercises
   - Focus on sound-letter relationships
   - Basic word recognition tasks
   - Frequent feedback and guidance

2. **Repetition-based Practice** (Level 2)
   - Structured practice sessions
   - Repeated exposure to target words
   - Pattern recognition exercises
   - Guided practice with support

3. **Visual-Audio Reinforcement** (Level 3)
   - Multi-modal learning activities
   - Audio-visual word matching
   - Interactive exercises
   - Moderate scaffolding

4. **Timed Decoding Challenges** (Level 4)
   - Speed-based reading tasks
   - Word recognition challenges
   - Time-pressure exercises
   - Reduced scaffolding

5. **Comprehension Extension** (Level 5)
   - Independent reading tasks
   - Comprehension questions
   - Context-based learning
   - Minimal scaffolding
""")

# Define intervention types dictionary
intervention_types = {
    1: "Phonological Support",
    2: "Repetition-based Practice",
    3: "Visual-Audio Reinforcement",
    4: "Timed Decoding Challenges",
    5: "Comprehension Extension"
}

# Performance by Intervention Level
st.subheader("Performance by Intervention Level")

# Calculate average performance for each level
level_performance = []
for level in range(1, 6):
    # Calculate which sessions correspond to each level
    # Level 1: Sessions 1-6
    # Level 2: Sessions 7-12
    # Level 3: Sessions 13-18
    # Level 4: Sessions 19-24
    # Level 5: Sessions 25-32
    start_session = (level - 1) * 6 + 1
    end_session = min(32, level * 6)
    
    # Calculate average metrics for these sessions
    level_data = filtered_df[filtered_df['Dyslexia'] == 1]  # Focus on dyslexic students
    
    # Calculate averages for each metric
    avg_accuracy = 0
    avg_score = 0
    avg_missrate = 0
    session_count = 0
    
    for session in range(start_session, end_session + 1):
        if f'Accuracy{session}' in level_data.columns:
            avg_accuracy += level_data[f'Accuracy{session}'].mean()
            avg_score += level_data[f'Score{session}'].mean()
            avg_missrate += level_data[f'Missrate{session}'].mean()
            session_count += 1
    
    if session_count > 0:
        avg_accuracy /= session_count
        avg_score /= session_count
        avg_missrate /= session_count
    
    level_performance.append({
        'Level': level,
        'Intervention Type': intervention_types[level],
        'Avg Accuracy': avg_accuracy,
        'Avg Score': avg_score,
        'Avg Miss Rate': avg_missrate
    })

level_df = pd.DataFrame(level_performance)

# Format numeric columns
numeric_columns = ['Avg Accuracy', 'Avg Score', 'Avg Miss Rate']
for col in numeric_columns:
    level_df[col] = level_df[col].round(3)

# Display level performance with fixed height
st.dataframe(
    level_df,
    use_container_width=True,
    height=150,  # Fixed height to prevent shaking
    hide_index=True
)

# Learning Progress Metrics
st.subheader("Learning Progress Metrics")

# Calculate learning progress indicators
progress_metrics = []
for _, row in filtered_df.iterrows():
    # Count improvements and regressions
    improvements = 0
    regressions = 0
    for i in range(2, 33):
        if row[f'Accuracy{i}'] > row[f'Accuracy{i-1}']:
            improvements += 1
        elif row[f'Accuracy{i}'] < row[f'Accuracy{i-1}']:
            regressions += 1
    
    # Calculate overall progress safely
    initial_accuracy = row['Accuracy1']
    final_accuracy = row['Accuracy32']
    
    if initial_accuracy == 0:
        if final_accuracy == 0:
            overall_progress = 0  # No improvement if both values are zero
        else:
            overall_progress = 100  # 100% improvement if starting from zero
    else:
        overall_progress = (final_accuracy - initial_accuracy) / initial_accuracy * 100
    
    progress_metrics.append({
        'Student ID': row.name,
        'Improvement Streak': improvements,
        'Regression Count': regressions,
        'Overall Progress': overall_progress
    })

progress_df = pd.DataFrame(progress_metrics)

# Format numeric columns
progress_df['Overall Progress'] = progress_df['Overall Progress'].round(2)

# Display progress metrics
st.dataframe(
    progress_df,
    use_container_width=True,
    hide_index=True
)

# Detailed Statistics
st.subheader("Detailed Statistics")

# Calculate statistics for each metric
stats = []
metrics = ['Accuracy', 'Score', 'Missrate']
for metric in metrics:
    for session in [1, 16, 32]:
        stats.append({
            'Metric': metric,
            'Session': session,
            'Mean': filtered_df[f'{metric}{session}'].mean(),
            'Std': filtered_df[f'{metric}{session}'].std(),
            'Min': filtered_df[f'{metric}{session}'].min(),
            'Max': filtered_df[f'{metric}{session}'].max()
        })

stats_df = pd.DataFrame(stats)

# Format numeric columns to 3 decimal places
numeric_columns = ['Mean', 'Std', 'Min', 'Max']
for col in numeric_columns:
    stats_df[col] = stats_df[col].round(3)

# Display the dataframe with fixed height and styling
st.dataframe(
    stats_df,
    use_container_width=True,
    height=300,  # Fixed height in pixels
    hide_index=True  # Hide the index column
) 