import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="DyAdapt - Analysis",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / "backend" / "data" / "dyslexia_data.csv"
    return pd.read_csv(data_path, sep=';')

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
    options=df['Dyslexia'].unique(),
    default=df['Dyslexia'].unique()
)

# Filter data
filtered_df = df[
    (df['Gender'].isin(selected_gender)) &
    (df['Dyslexia'].isin(selected_dyslexia))
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
    fig = px.box(filtered_df, y=['Accuracy1', 'Accuracy16', 'Accuracy32'],
                 title='Accuracy Distribution',
                 labels={'value': 'Accuracy', 'variable': 'Session'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(filtered_df, y=['Missrate1', 'Missrate16', 'Missrate32'],
                 title='Miss Rate Distribution',
                 labels={'value': 'Miss Rate', 'variable': 'Session'})
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = px.box(filtered_df, y=['Score1', 'Score16', 'Score32'],
                 title='Score Distribution',
                 labels={'value': 'Score', 'variable': 'Session'})
    st.plotly_chart(fig, use_container_width=True)

# Learning Progress Analysis
st.subheader("Learning Progress Analysis")

# Calculate improvement rates
def calculate_improvement(row, metric):
    initial_value = row[f'{metric}1']
    final_value = row[f'{metric}32']
    
    # Handle zero initial value
    if initial_value == 0:
        if final_value == 0:
            return 0  # No improvement if both values are zero
        return 100  # 100% improvement if starting from zero
    
    return (final_value - initial_value) / initial_value * 100

filtered_df['Accuracy_Improvement'] = filtered_df.apply(lambda x: calculate_improvement(x, 'Accuracy'), axis=1)
filtered_df['Score_Improvement'] = filtered_df.apply(lambda x: calculate_improvement(x, 'Score'), axis=1)
filtered_df['Missrate_Improvement'] = filtered_df.apply(lambda x: calculate_improvement(x, 'Missrate'), axis=1)

# Create improvement comparison plot
fig = px.box(filtered_df,
             y=['Accuracy_Improvement', 'Score_Improvement', 'Missrate_Improvement'],
             title='Improvement Rates by Metric',
             labels={'value': 'Improvement (%)', 'variable': 'Metric'})
st.plotly_chart(fig, use_container_width=True)

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