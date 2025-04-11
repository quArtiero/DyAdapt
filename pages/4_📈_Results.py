import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import requests
import json

st.set_page_config(
    page_title="DyAdapt - Results",
    page_icon="📈",
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

st.title("📈 Research Results")

# Load data
df = load_data()

# Overview Statistics
st.subheader("Overview Statistics")

# Calculate key metrics
total_students = len(df)
dyslexic_students = len(df[df['Dyslexia'] == 1])
non_dyslexic_students = len(df[df['Dyslexia'] == 0])
dyslexia_rate = (dyslexic_students / total_students * 100) if total_students > 0 else 0

# Display metrics in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Students", total_students)
with col2:
    st.metric("Dyslexic Students", dyslexic_students)
with col3:
    st.metric("Non-dyslexic Students", non_dyslexic_students)
with col4:
    st.metric("Dyslexia Rate", f"{dyslexia_rate:.1f}%")

# Model Performance
st.subheader("Model Performance")

# API endpoint
API_URL = "http://localhost:5002"

# Add train model button
if st.button("Train Model"):
    with st.spinner("Training model..."):
        try:
            response = requests.post(f'{API_URL}/train')
            if response.status_code == 200:
                st.success("Model trained successfully!")
                # Reload metrics
                model_path = Path(__file__).parent.parent / "backend" / "models" / "model_metrics.json"
                if model_path.exists():
                    with open(model_path, 'r') as f:
                        metrics = json.load(f)
            else:
                error_message = "Unknown error occurred"
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', error_message)
                except:
                    error_message = f"Server returned status code {response.status_code}"
                st.error(f"Error training model: {error_message}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the server. Please make sure the backend server is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# Load model performance metrics
model_path = Path(__file__).parent.parent / "backend" / "models" / "model_metrics.json"
metrics = {}
try:
    if model_path.exists():
        with open(model_path, 'r') as f:
            metrics = json.load(f)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        with col4:
            st.metric("F1 Score", f"{metrics.get('f1', 0):.1%}")
    else:
        st.info("Model performance metrics not available. Click 'Train Model' to generate metrics.")
except Exception as e:
    st.error(f"Error loading model metrics: {str(e)}")
    metrics = {}  # Ensure metrics is defined even if there's an error

# Learning Outcomes
st.subheader("Learning Outcomes")

# Calculate average performance by session for each group
dyslexic_performance = df[df['Dyslexia'] == 1].iloc[:, 4:].mean()
non_dyslexic_performance = df[df['Dyslexia'] == 0].iloc[:, 4:].mean()

# Create performance comparison plot
fig = go.Figure()

# Add dyslexic group
fig.add_trace(go.Scatter(
    x=list(range(1, 33)),
    y=dyslexic_performance[::3],  # Take every 3rd value (Accuracy)
    name='Dyslexic Students',
    line=dict(color='#1f77b4')
))

# Add non-dyslexic group
fig.add_trace(go.Scatter(
    x=list(range(1, 33)),
    y=non_dyslexic_performance[::3],  # Take every 3rd value (Accuracy)
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

# Intervention Effectiveness
st.subheader("Intervention Effectiveness")

# Calculate improvement rates
def calculate_improvement(row):
    initial_acc = row['Accuracy1']
    final_acc = row['Accuracy32']
    if initial_acc == 0:
        return 100 if final_acc > 0 else 0
    return ((final_acc - initial_acc) / initial_acc) * 100

df['Improvement'] = df.apply(calculate_improvement, axis=1)

# Create improvement comparison plot
# Boxplot with capped y-axis and mean indicator
fig = px.box(df,
             x='Dyslexia',
             y='Improvement',
             title='Distribution of Accuracy Improvement by Dyslexia Status',
             labels={'Dyslexia': 'Dyslexic Status', 'Improvement': 'Improvement Rate (%)'},
             color='Dyslexia',
             color_discrete_map={0: '#2ca02c', 1: '#1f77b4'})

fig.update_yaxes(range=[0, 500])  # Cap y-axis for clarity
fig.update_traces(boxmean=True)   # Add mean and std dev

st.plotly_chart(fig, use_container_width=True)

# Summary Table
st.subheader("Summary Statistics")

# Calculate summary statistics
summary_stats = {
    'Metric': ['Total Students', 'Dyslexic Students', 'Non-dyslexic Students', 
               'Average Improvement (Dyslexic)', 'Average Improvement (Non-dyslexic)',
               'Model Accuracy', 'Model Precision', 'Model Recall', 'Model F1 Score'],
    'Value': [
        total_students,
        dyslexic_students,
        non_dyslexic_students,
        f"{df[df['Dyslexia'] == 1]['Improvement'].mean():.1f}%",
        f"{df[df['Dyslexia'] == 0]['Improvement'].mean():.1f}%",
        f"{metrics.get('accuracy', 0):.1%}",
        f"{metrics.get('precision', 0):.1%}",
        f"{metrics.get('recall', 0):.1%}",
        f"{metrics.get('f1', 0):.1%}"
    ]
}

summary_df = pd.DataFrame(summary_stats)

# Format numeric columns
numeric_columns = ['Value']
for col in numeric_columns:
    if col in summary_df.columns:
        summary_df[col] = summary_df[col].astype(str)

# Display summary statistics with fixed height
st.dataframe(
    summary_df,
    use_container_width=True,
    height=300,  # Fixed height to prevent shaking
    hide_index=True
)

# Download Results
st.subheader("Download Results")

# Create CSV data
csv_data = summary_df.to_csv(index=False)

# Add download button
st.download_button(
    label="Download Results Summary",
    data=csv_data,
    file_name="research_results.csv",
    mime="text/csv"
) 