import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, make_scorer, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
from pathlib import Path
import datetime

class DyslexiaPredictor:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.feature_importance = None
        self.metrics = {}
        
    def save_metrics(self, metrics_dict):
        """Save model metrics to a JSON file with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = Path('backend/output/model_metrics.json')
        
        # Load existing metrics if file exists
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}
        
        # Add new metrics with timestamp
        all_metrics[timestamp] = metrics_dict
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        print(f"\nMetrics saved to {metrics_file}")
        
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
        
        # Convert Age to numeric and handle missing values
        if 'Age' in X.columns:
            X['Age'] = pd.to_numeric(X['Age'], errors='coerce')
            X['Age'].fillna(X['Age'].median(), inplace=True)
        
        return X, y
    
    def select_features(self, X, y, k=50):
        """Enhanced feature selection using multiple methods."""
        # ANOVA F-test
        f_selector = SelectKBest(score_func=f_classif, k=k)
        f_scores = f_selector.fit_transform(X, y)
        f_selected_features = X.columns[f_selector.get_support()].tolist()
        f_feature_scores = f_selector.scores_[f_selector.get_support()]
        
        # Mutual Information
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        mi_scores = mi_selector.fit_transform(X, y)
        mi_selected_features = X.columns[mi_selector.get_support()].tolist()
        mi_feature_scores = mi_selector.scores_[mi_selector.get_support()]
        
        # Combine scores
        combined_scores = {}
        for feature in set(f_selected_features + mi_selected_features):
            f_score = f_selector.scores_[list(X.columns).index(feature)]
            mi_score = mi_selector.scores_[list(X.columns).index(feature)]
            combined_scores[feature] = (f_score + mi_score) / 2
        
        # Select top k features based on combined scores
        selected_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        self.feature_importance = dict(selected_features)
        
        # Print top features
        print("\nTop Features by Importance:")
        for feature, score in selected_features[:10]:
            print(f"{feature}: {score:.2f}")
        
        # Create final selector
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        return X_selected
    
    def optimize_parameters(self, X_train, y_train):
        """Optimize model parameters using grid search."""
        # Define parameter grid
        param_grid = {
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly'],
                'degree': [2, 3],  # for poly kernel
                'class_weight': ['balanced', None]
            },
            'rf': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced', 'balanced_subsample']
            }
        }
        
        # Create base classifiers
        svm = SVC(probability=True, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        
        # Perform grid search for each classifier
        svm_grid = GridSearchCV(svm, param_grid['svm'], cv=5, scoring='f1_macro', n_jobs=-1)
        rf_grid = GridSearchCV(rf, param_grid['rf'], cv=5, scoring='f1_macro', n_jobs=-1)
        
        # Fit grid search
        svm_grid.fit(X_train, y_train)
        rf_grid.fit(X_train, y_train)
        
        print("\nBest SVM parameters:", svm_grid.best_params_)
        print("Best RF parameters:", rf_grid.best_params_)
        
        return svm_grid.best_estimator_, rf_grid.best_estimator_
    
    def train_model(self, X, y):
        """Train an ensemble model with optimized parameters."""
        # Select features
        X_selected = self.select_features(X, y)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Print class distribution
        print("\nClass distribution:")
        print(pd.Series(y_train).value_counts())
        
        # Apply SMOTE with higher ratio
        smote = SMOTE(sampling_strategy=0.8, random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Optimize and get best models
        best_svm, best_rf = self.optimize_parameters(X_train_balanced, y_train_balanced)
        
        # Create voting classifier
        self.model = VotingClassifier(
            estimators=[
                ('svm', best_svm),
                ('rf', best_rf)
            ],
            voting='soft'  # Use probability estimates
        )
        
        # Train ensemble
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(self.model, X_train_balanced, y_train_balanced, cv=5)
        print("\nCross-validation scores:", cv_scores)
        print("Mean CV score:", cv_scores.mean())
        
        # Evaluate on training set
        y_train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        print(f"\nTraining Accuracy: {train_accuracy:.3f}")
        print("\nTraining Classification Report:")
        print(classification_report(y_train, y_train_pred))
        
        # Evaluate on test set
        y_test_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        print(f"\nTest Accuracy: {test_accuracy:.3f}")
        print("\nTest Classification Report:")
        print(classification_report(y_test, y_test_pred))
        
        # Save metrics
        metrics = {
            'dataset_size': len(X),
            'class_distribution': {
                'non_dyslexic': int(sum(y == 0)),
                'dyslexic': int(sum(y == 1))
            },
            'feature_importance': self.feature_importance,
            'cross_validation': {
                'scores': cv_scores.tolist(),
                'mean_score': float(cv_scores.mean()),
                'std_score': float(cv_scores.std())
            },
            'training_metrics': {
                'accuracy': float(train_accuracy),
                'classification_report': train_report
            },
            'test_metrics': {
                'accuracy': float(test_accuracy),
                'classification_report': test_report
            },
            'model_parameters': {
                'svm': best_svm.get_params(),
                'rf': best_rf.get_params()
            }
        }
        self.save_metrics(metrics)
        
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