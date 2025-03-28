from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

app = Flask(__name__)

# Load the model and components
model_path = Path('backend/models/dyslexia_classifier.pkl')
predictor = joblib.load(model_path)

class AdaptiveLearningSystem:
    def __init__(self):
        self.students = {}
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)
    
    def export_simulation_logs(self):
        """Export simulation logs to CSV for analysis."""
        logs = []
        
        for student_id, student_data in self.students.items():
            for session in student_data['sessions']:
                log_entry = {
                    'student_id': student_id,
                    'session': session.get('session_number', 0),
                    'accuracy': session.get('accuracy', 0),
                    'miss_rate': session.get('miss_rate', 0),
                    'score': session.get('score', 0),
                    'intervention_level': session.get('level', 1),
                    'intervention_type': session.get('intervention_type', ''),
                    'fatigue_factor': session.get('fatigue_factor', 1.0),
                    'dyslexia_status': student_data.get('dyslexia_status', 'Unknown')
                }
                logs.append(log_entry)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(logs)
        output_path = self.output_dir / "simulation_logs.csv"
        df.to_csv(output_path, index=False)
        return str(output_path)

    def plot_learning_curves(self, student_id: str) -> Dict[str, Any]:
        """Generate actual vs. ideal learning curves for a student."""
        student = self.students.get(student_id)
        if not student:
            raise ValueError(f"Student {student_id} not found")
        
        sessions = range(1, len(student['sessions']) + 1)
        
        # Get actual intervention levels
        actual_levels = [session.get('level', 1) for session in student['sessions']]
        
        # Generate ideal curve (sigmoid)
        x = np.array(sessions)
        ideal_levels = 1 + 4 * (1 / (1 + np.exp(-0.2 * (x - len(sessions)/2))))
        
        # Calculate accuracy progression
        actual_accuracy = [session.get('accuracy', 0) for session in student['sessions']]
        
        return {
            'sessions': list(sessions),
            'actual_levels': actual_levels,
            'ideal_levels': ideal_levels.tolist(),
            'accuracy': actual_accuracy
        }

    def compare_groups(self) -> Dict[str, Any]:
        """Compare learning patterns between dyslexic and non-dyslexic students."""
        dyslexic_students = [s for s in self.students.values() if s.get('dyslexia_status') == 'Yes']
        non_dyslexic_students = [s for s in self.students.values() if s.get('dyslexia_status') == 'No']
        
        def calculate_group_metrics(group):
            if not group:
                return {'accuracy': [], 'levels': []}
            
            max_sessions = max(len(s['sessions']) for s in group)
            accuracy_by_session = [[] for _ in range(max_sessions)]
            levels_by_session = [[] for _ in range(max_sessions)]
            
            for student in group:
                for i, session in enumerate(student['sessions']):
                    accuracy_by_session[i].append(session.get('accuracy', 0))
                    levels_by_session[i].append(session.get('level', 1))
            
            return {
                'accuracy': [np.mean(session) for session in accuracy_by_session],
                'levels': [np.mean(session) for session in levels_by_session]
            }
        
        return {
            'dyslexic': calculate_group_metrics(dyslexic_students),
            'non_dyslexic': calculate_group_metrics(non_dyslexic_students)
        }

# Initialize the system
system = AdaptiveLearningSystem()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess data
        X = df.copy()
        
        # Convert categorical columns to numeric
        categorical_cols = ['Gender', 'Nativelang', 'Otherlang']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = (X[col] == 'Yes').astype(int)
        
        # Convert Age to numeric and handle missing values
        if 'Age' in X.columns:
            X['Age'] = pd.to_numeric(X['Age'], errors='coerce')
            X['Age'].fillna(X['Age'].median(), inplace=True)
        
        # Select features
        X_selected = predictor.feature_selector.transform(X)
        
        # Scale features
        X_scaled = predictor.scaler.transform(X_selected)
        
        # Make prediction
        prediction = predictor.model.predict(X_scaled)
        probability = predictor.model.predict_proba(X_scaled)
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),  # Probability of being dyslexic
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    })

@app.route('/export_logs', methods=['POST'])
def export_logs():
    try:
        output_path = system.export_simulation_logs()
        return jsonify({'status': 'success', 'output_path': output_path})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/learning_curves/<student_id>', methods=['GET'])
def get_learning_curves(student_id):
    try:
        curves = system.plot_learning_curves(student_id)
        return jsonify(curves)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/compare_groups', methods=['GET'])
def get_group_comparison():
    try:
        comparison = system.compare_groups()
        return jsonify(comparison)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def adjust_difficulty(self, student_id: str, session_data: Dict[str, float]) -> Dict[str, Any]:
    """Adjust difficulty based on performance and learning history."""
    # Get current student state
    student = self.students.get(student_id)
    if not student:
        raise ValueError(f"Student {student_id} not found")
    
    # Get current level
    current_level = student.get('current_level', 1)
    
    # Calculate performance metrics
    accuracy = session_data.get('accuracy', 0)
    miss_rate = session_data.get('miss_rate', 0)
    score = session_data.get('score', 0)
    
    # Add performance noise and fatigue
    if np.random.rand() < 0.2:  # 20% chance of underperformance
        accuracy -= np.random.uniform(0.05, 0.1)
        accuracy = max(0, min(1, accuracy))  # Keep between 0 and 1
    
    # Add session fatigue (performance decreases with more sessions)
    session_count = student.get('session_count', 0)
    fatigue_factor = 1 - (session_count * 0.01)  # 1% decrease per session
    fatigue_factor = max(0.8, fatigue_factor)  # Cap at 20% reduction
    accuracy *= fatigue_factor
    
    # Define intervention types
    intervention_types = {
        1: "phonological support",
        2: "repetition-based practice",
        3: "visual-audio reinforcement",
        4: "timed decoding challenges",
        5: "comprehension extension tasks"
    }
    
    # Calculate improvement streak and regression count
    improvement_streak = student.get('improvement_streak', 0)
    regression_count = student.get('regression_count', 0)
    
    # Determine adjustment based on performance
    if accuracy >= 0.8 and miss_rate <= 0.2:
        # Good performance - increase difficulty
        new_level = min(5, current_level + 1)
        improvement_streak += 1
        regression_count = 0
    elif accuracy <= 0.6 or miss_rate >= 0.4:
        # Poor performance - decrease difficulty
        new_level = max(1, current_level - 1)
        improvement_streak = 0
        regression_count += 1
    else:
        # Maintain current level
        new_level = current_level
    
    # Update student state
    student['current_level'] = new_level
    student['session_count'] = session_count + 1
    student['improvement_streak'] = improvement_streak
    student['regression_count'] = regression_count
    student['last_intervention'] = intervention_types[new_level]
    
    # Store session data
    student['sessions'].append({
        'accuracy': accuracy,
        'miss_rate': miss_rate,
        'score': score,
        'level': new_level,
        'intervention_type': intervention_types[new_level],
        'fatigue_factor': fatigue_factor
    })
    
    # Update student in database
    self.students[student_id] = student
    
    return {
        'new_level': new_level,
        'intervention_type': intervention_types[new_level],
        'improvement_streak': improvement_streak,
        'regression_count': regression_count,
        'fatigue_factor': fatigue_factor,
        'session_count': session_count + 1
    }

if __name__ == '__main__':
    app.run(debug=True, port=5000) 