import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd
import numpy as np

def load_session_data(file_path: str) -> list:
    """Load session data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_session_plots(student_data: dict) -> go.Figure:
    """Create interactive plots for a student's learning sessions."""
    # Convert session data to DataFrame
    sessions_df = pd.DataFrame(student_data['sessions'])
    
    # Create subplots with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Accuracy and Intervention Level Over Time',
            'Response Time Distribution',
            'Performance Categories',
            'Learning Recommendations'
        ),
        specs=[
            [{"type": "xy", "secondary_y": True}, {"type": "box"}],
            [{"type": "pie"}, {"type": "bar"}]
        ]
    )
    
    # Plot 1: Accuracy and Intervention Level
    fig.add_trace(
        go.Scatter(
            x=sessions_df['session_id'],
            y=sessions_df['accuracy'],
            name='Accuracy',
            line=dict(color='blue')
        ),
        row=1, col=1,
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=sessions_df['session_id'],
            y=sessions_df['intervention_level'],
            name='Intervention Level',
            line=dict(color='red')
        ),
        row=1, col=1,
        secondary_y=True
    )
    
    # Plot 2: Response Time Distribution
    fig.add_trace(
        go.Box(
            y=sessions_df['response_time'],
            name='Response Time',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ),
        row=1, col=2
    )
    
    # Plot 3: Performance Categories
    performance_counts = sessions_df['performance_category'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=performance_counts.index,
            values=performance_counts.values,
            hole=0.4,
            name='Performance Categories'
        ),
        row=2, col=1
    )
    
    # Plot 4: Learning Recommendations
    all_recommendations = [rec for recs in sessions_df['recommendations'] for rec in recs]
    rec_counts = pd.Series(all_recommendations).value_counts()
    fig.add_trace(
        go.Bar(
            x=rec_counts.index,
            y=rec_counts.values,
            name='Recommendations'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        title_text=f"Learning Progress Analysis - Student {student_data['student_id']}",
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Session Number", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Intervention Level", row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Response Time (seconds)", row=1, col=2)
    fig.update_xaxes(title_text="Recommendations", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    return fig

def create_feedback_table(student_data: dict) -> pd.DataFrame:
    """Create a detailed feedback table for a student's sessions."""
    sessions_df = pd.DataFrame(student_data['sessions'])
    
    # Calculate summary statistics
    summary_stats = {
        'Average Accuracy': f"{sessions_df['accuracy'].mean():.3f}",
        'Average Response Time': f"{sessions_df['response_time'].mean():.2f}s",
        'Final Intervention Level': f"{sessions_df['intervention_level'].iloc[-1]}/5",
        'Most Common Performance': sessions_df['performance_category'].mode().iloc[0],
        'Progress Trend': 'Improving' if sessions_df['accuracy'].iloc[-1] > sessions_df['accuracy'].iloc[0] else 'Stable'
    }
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([summary_stats])
    
    # Save to CSV
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    summary_df.to_csv(output_dir / f"student_{student_data['student_id']}_summary.csv", index=False)
    
    return summary_df

def main():
    # Load session data
    data_path = Path(__file__).parent.parent / "output" / "student_results.json"
    session_data = load_session_data(data_path)
    
    # Create plots and tables for each student
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    for student in session_data:
        # Create and save plots
        fig = create_session_plots(student)
        fig.write_html(output_dir / f"student_{student['student_id']}_plots.html")
        
        # Create and save feedback table
        summary_df = create_feedback_table(student)
        print(f"\nSummary for Student {student['student_id']}:")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main() 