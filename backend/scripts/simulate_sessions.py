import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path

class SessionSimulator:
    def __init__(self):
        self.base_accuracy = {
            0: 0.7,  # Non-dyslexic students
            1: 0.5   # Dyslexic students
        }
        self.base_response_time = {
            0: 1.5,  # Non-dyslexic students
            1: 2.5   # Dyslexic students
        }
        self.learning_rate = {
            0: 0.02,  # Non-dyslexic students
            1: 0.01   # Dyslexic students
        }
    
    def adjust_intervention(self, 
                          current_level: int,
                          accuracy: float,
                          response_time: float,
                          previous_feedback: str) -> Dict[str, Any]:
        """
        Adjust intervention level based on performance metrics.
        Returns dict with new level and feedback.
        """
        # Calculate performance score (weighted combination)
        perf_score = 0.7 * accuracy - 0.3 * (response_time / 5.0)
        
        # Determine feedback based on performance
        if perf_score > 0.7:
            feedback = "harder"
            new_level = min(5, current_level + 1)
        elif perf_score < 0.3:
            feedback = "easier"
            new_level = max(1, current_level - 1)
        else:
            feedback = "maintain"
            new_level = current_level
        
        # Add explanation for the decision
        explanation = self._generate_feedback_explanation(
            accuracy, response_time, perf_score, feedback
        )
        
        return {
            "new_level": new_level,
            "feedback": feedback,
            "explanation": explanation
        }
    
    def _generate_feedback_explanation(self,
                                    accuracy: float,
                                    response_time: float,
                                    perf_score: float,
                                    feedback: str) -> str:
        """Generate human-readable explanation for feedback decision."""
        if feedback == "harder":
            return f"High performance (accuracy: {accuracy:.2f}, response time: {response_time:.1f}s) suggests student is ready for more challenging content."
        elif feedback == "easier":
            return f"Current difficulty may be too high (accuracy: {accuracy:.2f}, response time: {response_time:.1f}s). Reducing complexity to build confidence."
        else:
            return f"Performance is balanced (accuracy: {accuracy:.2f}, response time: {response_time:.1f}s). Maintaining current level."
    
    def simulate_session(self,
                        student_type: int,
                        current_level: int,
                        session_number: int) -> Dict[str, Any]:
        """Simulate a single learning session."""
        # Base performance with learning improvement
        base_acc = self.base_accuracy[student_type]
        base_rt = self.base_response_time[student_type]
        lr = self.learning_rate[student_type]
        
        # Add learning progress over sessions
        accuracy = base_acc + (lr * session_number)
        accuracy = max(0.1, min(0.9, accuracy))  # Clamp between 0.1 and 0.9
        
        # Response time with some randomness
        response_time = np.random.normal(base_rt, 0.5)
        response_time = max(0.5, response_time)  # Minimum 0.5 seconds
        
        # Adjust intervention based on performance
        adjustment = self.adjust_intervention(
            current_level, accuracy, response_time, "maintain"
        )
        
        return {
            "session_id": session_number,
            "accuracy": round(accuracy, 3),
            "response_time": round(response_time, 2),
            "intervention_level": adjustment["new_level"],
            "feedback": adjustment["feedback"],
            "explanation": adjustment["explanation"]
        }
    
    def simulate_student_path(self,
                            student_id: int,
                            student_type: int,
                            n_sessions: int = 10) -> Dict[str, Any]:
        """Simulate complete learning path for a student."""
        sessions = []
        current_level = 3  # Start at middle level
        
        for session in range(1, n_sessions + 1):
            session_data = self.simulate_session(
                student_type, current_level, session
            )
            current_level = session_data["intervention_level"]
            sessions.append(session_data)
        
        return {
            "student_id": student_id,
            "predicted_dyslexia": student_type,
            "sessions": sessions
        }

def main():
    # Example usage
    simulator = SessionSimulator()
    
    # Simulate paths for both types of students
    results = []
    for student_id, student_type in enumerate([0, 1]):
        student_path = simulator.simulate_student_path(
            student_id + 1, student_type
        )
        results.append(student_path)
    
    # Save example results
    output_path = Path(__file__).parent.parent / "output" / "example_sessions.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 