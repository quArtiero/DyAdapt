from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from imblearn.over_sampling import SMOTE
import logging
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model and components
try:
    model_path = Path(__file__).parent / 'models' / 'dyslexia_classifier.pkl'
    logger.info(f"Loading model from: {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    predictor = joblib.load(model_path)
    logger.info("Model loaded successfully")
    
    # Verify model components
    if not hasattr(predictor, 'predict'):
        raise AttributeError("Loaded model does not have predict method")
        
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    predictor = None

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
    if predictor is None:
        logger.error("Prediction failed: Model not loaded")
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
        
    try:
        # Get data from request
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")
        
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
            X['Age'] = X['Age'].fillna(X['Age'].median())
        
        # Select features
        X_selected = predictor.feature_selector.transform(X)
        
        # Scale features
        X_scaled = predictor.scaler.transform(X_selected)
        
        # Make prediction
        prediction = predictor.model.predict(X_scaled)
        probability = predictor.model.predict_proba(X_scaled)
        
        logger.info(f"Prediction: {prediction[0]}, Probability: {probability[0][1]}")
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),  # Probability of being dyslexic
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    if predictor is None:
        logger.error("Health check failed: Model not loaded")
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
        
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
        # Load the data
        data_path = Path(__file__).parent / "data" / "dyslexia_data.csv"
        df = pd.read_csv(data_path, sep=';')
        
        # Get student data
        student_data = df.iloc[int(student_id)]
        
        # Generate learning curve data
        sessions = list(range(1, 33))
        accuracy = [student_data[f'Accuracy{i}'] for i in range(1, 33)]
        
        # Generate ideal curve (sigmoid)
        x = np.array(sessions)
        ideal_levels = 1 + 4 * (1 / (1 + np.exp(-0.2 * (x - len(sessions)/2))))
        
        # Calculate actual levels based on performance
        actual_levels = []
        for acc in accuracy:
            if acc >= 0.8:
                level = 5
            elif acc >= 0.6:
                level = 4
            elif acc >= 0.4:
                level = 3
            elif acc >= 0.2:
                level = 2
            else:
                level = 1
            actual_levels.append(level)
        
        return jsonify({
            'sessions': sessions,
            'actual_levels': actual_levels,
            'ideal_levels': ideal_levels.tolist(),
            'accuracy': accuracy
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/compare_groups', methods=['GET'])
def get_group_comparison():
    try:
        # Load the data
        data_path = Path(__file__).parent / "data" / "dyslexia_data.csv"
        print(f"Loading data from: {data_path}")
        
        if not data_path.exists():
            print(f"Error: Data file not found at {data_path}")
            return jsonify({'status': 'error', 'message': 'Data file not found'}), 404
            
        df = pd.read_csv(data_path, sep=';')
        print(f"Loaded {len(df)} rows of data")
        
        # Convert Dyslexia column to numeric (Yes=1, No=0)
        df['Dyslexia'] = (df['Dyslexia'] == 'Yes').astype(int)
        print(f"Dyslexic students: {df['Dyslexia'].sum()}, Non-dyslexic: {len(df) - df['Dyslexia'].sum()}")
        
        # Calculate average accuracy by session for each group
        dyslexic_data = df[df['Dyslexia'] == 1]
        non_dyslexic_data = df[df['Dyslexia'] == 0]
        
        # Calculate average accuracy for each session
        dyslexic_accuracy = []
        non_dyslexic_accuracy = []
        
        for i in range(1, 33):
            acc_col = f'Accuracy{i}'
            if acc_col not in df.columns:
                print(f"Warning: Column {acc_col} not found in data")
                continue
                
            dyslexic_acc = dyslexic_data[acc_col].mean()
            non_dyslexic_acc = non_dyslexic_data[acc_col].mean()
            
            dyslexic_accuracy.append(float(dyslexic_acc))
            non_dyslexic_accuracy.append(float(non_dyslexic_acc))
        
        # Calculate average intervention level for each session
        dyslexic_levels = [min(5, 1 + (i-1)/8) for i in range(1, 33)]  # Slower progression
        non_dyslexic_levels = [min(5, 1 + (i-1)/6) for i in range(1, 33)]  # Faster progression
        
        response = {
            'dyslexic': {
                'accuracy': dyslexic_accuracy,
                'levels': dyslexic_levels
            },
            'non_dyslexic': {
                'accuracy': non_dyslexic_accuracy,
                'levels': non_dyslexic_levels
            }
        }
        
        print("Successfully generated group comparison data")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in compare_groups: {str(e)}")
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

def train_model():
    """Train the dyslexia detection model."""
    try:
        # Load and preprocess data
        data_path = Path(__file__).parent / "data" / "dyslexia_data.csv"
        df = pd.read_csv(data_path, sep=';')
        
        # Convert categorical columns to numeric
        categorical_cols = ['Gender', 'Nativelang', 'Otherlang']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = (df[col] == 'Yes').astype(int)
        
        # Convert Dyslexia column to numeric (Yes=1, No=0)
        df['Dyslexia'] = (df['Dyslexia'] == 'Yes').astype(int)
        
        # Convert Age to numeric and handle missing values
        if 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            df['Age'].fillna(df['Age'].median(), inplace=True)
        
        # Prepare features and target
        X = df.drop(['Dyslexia'], axis=1)  # Keep all features except Dyslexia
        y = df['Dyslexia']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
        
        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Train models on balanced data
        svm.fit(X_train_balanced, y_train_balanced)
        rf.fit(X_train_balanced, y_train_balanced)
        
        # Make predictions
        svm_pred = svm.predict(X_test_scaled)
        rf_pred = rf.predict(X_test_scaled)
        
        # Combine predictions (ensemble)
        ensemble_pred = (svm_pred + rf_pred) / 2
        ensemble_pred = (ensemble_pred > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, ensemble_pred)
        precision = precision_score(y_test, ensemble_pred)
        recall = recall_score(y_test, ensemble_pred)
        f1 = f1_score(y_test, ensemble_pred)
        
        # Save metrics to JSON file
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        metrics_path = Path(__file__).parent / "models" / "model_metrics.json"
        metrics_path.parent.mkdir(exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        # Save models
        models_path = Path(__file__).parent / "models"
        models_path.mkdir(exist_ok=True)
        
        joblib.dump(svm, models_path / "svm_model.joblib")
        joblib.dump(rf, models_path / "rf_model.joblib")
        joblib.dump(scaler, models_path / "scaler.joblib")
        
        return True, "Model trained successfully"
    except Exception as e:
        print(f"Error in train_model: {str(e)}")  # Add debug logging
        return False, str(e)

@app.route('/train', methods=['POST'])
def train():
    """Train the model and return the results."""
    try:
        success, message = train_model()
        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002) 