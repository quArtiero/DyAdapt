import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import joblib
import json
from pathlib import Path

class DyslexiaPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def load_data(self, file_path):
        """Load and preprocess the dyslexia dataset."""
        # Read CSV with semicolon separator
        df = pd.read_csv(file_path, sep=';')
        
        # Print dataset info
        print("\nDataset Info:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        print("\nColumn names:")
        print(df.columns.tolist())
        
        # Convert Dyslexia column to numeric (assuming 'Yes'/'No' format)
        df['Dyslexia'] = (df['Dyslexia'] == 'Yes').astype(int)
        
        # Split features and target
        X = df.drop('Dyslexia', axis=1)
        y = df['Dyslexia']
        
        # Convert categorical columns to numeric
        categorical_cols = ['Gender', 'Nativelang', 'Otherlang']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = (X[col] == 'Yes').astype(int)
        
        # Convert Age to numeric
        if 'Age' in X.columns:
            X['Age'] = pd.to_numeric(X['Age'], errors='coerce')
        
        return X, y
    
    def select_features(self, X, y, k=50):
        """Select the most important features using ANOVA F-test."""
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        print("\nSelected Features:")
        print(selected_features)
        
        return X_selected
    
    def train_model(self, X, y):
        """Train the SVM classifier with class balancing."""
        # Select features first
        X_selected = self.select_features(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Train SVM with balanced classes
        self.model = SVC(kernel='rbf', probability=True, class_weight='balanced')
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def simulate_sessions(self, student_data, n_sessions=10):
        """Simulate learning sessions for a student."""
        sessions = []
        current_level = 3  # Start at middle level
        
        # Define learning recommendations based on performance
        recommendations = {
            "high": [
                "Consider introducing more complex reading materials",
                "Focus on advanced comprehension strategies",
                "Practice multi-step problem solving"
            ],
            "medium": [
                "Maintain current reading level with varied content",
                "Practice both speed and accuracy",
                "Include regular comprehension checks"
            ],
            "low": [
                "Break down complex tasks into smaller steps",
                "Provide more visual aids and examples",
                "Focus on building foundational skills"
            ]
        }
        
        for session in range(n_sessions):
            # Simulate performance based on student's predicted dyslexia status
            base_accuracy = 0.7 if student_data['predicted_dyslexia'] == 0 else 0.5
            accuracy = np.random.normal(base_accuracy, 0.1)
            accuracy = max(0.1, min(0.9, accuracy))  # Clamp between 0.1 and 0.9
            
            response_time = np.random.normal(2.0, 0.5)  # Simulated response time in seconds
            
            # Determine performance category and feedback
            if accuracy > 0.8:
                feedback = "harder"
                current_level = min(5, current_level + 1)
                performance_category = "high"
                explanation = f"High performance (accuracy: {accuracy:.2f}) indicates strong comprehension. "
                explanation += f"Current level: {current_level}/5. "
                explanation += f"Response time: {response_time:.1f}s."
            elif accuracy < 0.4:
                feedback = "easier"
                current_level = max(1, current_level - 1)
                performance_category = "low"
                explanation = f"Current difficulty may be too high (accuracy: {accuracy:.2f}). "
                explanation += f"Reducing complexity to build confidence. "
                explanation += f"Current level: {current_level}/5. "
                explanation += f"Response time: {response_time:.1f}s."
            else:
                feedback = "maintain"
                performance_category = "medium"
                explanation = f"Performance is balanced (accuracy: {accuracy:.2f}). "
                explanation += f"Maintaining current level: {current_level}/5. "
                explanation += f"Response time: {response_time:.1f}s."
            
            # Select random recommendations based on performance category
            selected_recommendations = np.random.choice(
                recommendations[performance_category],
                size=min(2, len(recommendations[performance_category])),
                replace=False
            ).tolist()
            
            sessions.append({
                "session_id": session + 1,
                "accuracy": round(accuracy, 3),
                "response_time": round(response_time, 2),
                "intervention_level": current_level,
                "feedback": feedback,
                "explanation": explanation,
                "recommendations": selected_recommendations,
                "performance_category": performance_category
            })
        
        return sessions
    
    def process_students(self, X):
        """Process all students and generate their learning paths."""
        # Transform features using the trained feature selector
        X_selected = self.feature_selector.transform(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        results = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            student_data = {
                "student_id": idx + 1,
                "predicted_dyslexia": int(pred),
                "confidence": float(max(prob)),
                "sessions": self.simulate_sessions({"predicted_dyslexia": pred})
            }
            results.append(student_data)
        
        return results

def main():
    # Initialize predictor
    predictor = DyslexiaPredictor()
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / "Dyt-desktop.csv"
    X, y = predictor.load_data(data_path)
    
    # Train model
    predictor.train_model(X, y)
    
    # Process all students
    results = predictor.process_students(X)
    
    # Save results
    output_path = Path(__file__).parent.parent / "output" / "student_results.json"
    output_path.parent.mkdir(exist_ok=True)  # Create output directory if it doesn't exist
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model and scaler
    model_path = Path(__file__).parent.parent / "models"
    model_path.mkdir(exist_ok=True)  # Create models directory if it doesn't exist
    joblib.dump(predictor.model, model_path / "dyslexia_classifier.pkl")
    joblib.dump(predictor.scaler, model_path / "scaler.pkl")
    joblib.dump(predictor.feature_selector, model_path / "feature_selector.pkl")

if __name__ == "__main__":
    main() 