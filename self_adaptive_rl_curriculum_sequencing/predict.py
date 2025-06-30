import os
import torch
import numpy as np
from models.self_adaptive_agent import SelfAdaptiveAgent
from utils.student_env import StudentLearningEnv
from config.config import rl_config, curriculum_config, self_adaptive_config, experiment_config

class DifficultyPredictor:
    def __init__(self, model_path):
        """Initialize the predictor with a trained model."""
        # Create a dummy environment to get observation space dimensions
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "preprocessed_kt_data.csv")
        self.env = StudentLearningEnv(data_path, curriculum_config, split='test')
        
        # Initialize agent
        self.agent = SelfAdaptiveAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            config=rl_config,
            curriculum_config=curriculum_config,
            self_adaptive_config=self_adaptive_config
        )
        
        # Load the trained model
        self.load_model(model_path)
        self.agent.policy_net.eval()  # Set to evaluation mode
    
    def load_model(self, model_path):
        """Load the trained model weights."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        
        # Load target network if available
        if 'target_net_state_dict' in checkpoint:
            self.agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    
    def predict_difficulty(self, student_data):
        """
        Predict the difficulty level for a given student's state.
        
        Args:
            student_data (dict): Dictionary containing student features
            
        Returns:
            dict: Predicted difficulty level and confidence scores
        """
        # Convert student data to state vector
        state = self._prepare_state(student_data)
        
        # Get Q-values for all actions
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(experiment_config.device)
            q_values = self.agent.policy_net(state_tensor).squeeze().cpu().numpy()
        
        # Get predicted action (difficulty level)
        predicted_action = np.argmax(q_values)
        
        # Convert to difficulty level (0.1 to 1.0)
        difficulty = 0.1 + (predicted_action / 4.0) * 0.9
        
        # Calculate confidence scores (softmax of Q-values)
        confidence_scores = np.exp(q_values) / np.sum(np.exp(q_values))
        
        return {
            'difficulty_level': predicted_action,  # 0-4
            'difficulty_score': float(difficulty),  # 0.1-1.0
            'confidence_scores': {f'level_{i}': float(conf) for i, conf in enumerate(confidence_scores)}
        }
    
    def _prepare_state(self, student_data):
        """Convert student data dictionary to state vector."""
        # Create state vector with the same structure as in the environment
        state = np.zeros(self.env.observation_space.shape[0], dtype=np.float32)
        
        # Map student data to state vector
        # Note: Adjust these indices based on your state representation
        state[0] = student_data.get('score', 0.5)  # Normalized score (0-1)
        state[1] = student_data.get('time_spent', 0.5)  # Normalized time spent
        state[2] = student_data.get('total_attempts', 0.5)  # Normalized attempts
        state[3] = student_data.get('cumulative_passes', 0.5)  # Normalized passes
        state[4] = student_data.get('pass_rate', 0.5)  # Pass rate
        state[5] = student_data.get('mean_perception', 0.5)  # Mean perception
        state[6] = 0 if student_data.get('sex', 'Boy') == 'Boy' else 1  # Gender
        state[7] = 0.5  # Current difficulty (default)
        state[8] = 0.0  # Progress (0-1)
        state[9] = 0.5  # Recent performance (0-1)
        
        return state

def main():
    # Initialize predictor with the trained model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "model_20250630_222553", "best_model_ep20_reward10.78.pt")
    data_path = os.path.join(base_dir, "preprocessed_kt_data.csv")
    
    try:
        # Initialize the test environment
        print("Initializing test environment...")
        test_env = StudentLearningEnv(
            data_path=data_path,
            config=curriculum_config,  # Changed from curriculum_config to config
            split='test',
            seed=experiment_config.seed + 2  # Different seed than train/val
        )
        
        # Initialize predictor with the same state/action space as the environment
        predictor = DifficultyPredictor(model_path)
        print("Model loaded successfully!")
        
        # Run predictions on test data
        print("\nMaking predictions on test set...")
        test_students = test_env.data['student_id'].unique()
        print(f"Number of test students: {len(test_students)}")
        
        # Get predictions for first few test students as example
        num_examples = min(5, len(test_students))
        print(f"\nExample predictions for {num_examples} test students:")
        
        for i in range(num_examples):
            # Reset environment to get a test student
            state = test_env.reset()
            student_id = test_env.current_student_id
            
            # Get student data for prediction
            student_data = test_env.current_student_data.iloc[0].to_dict()
            
            # Make prediction
            prediction = predictor.predict_difficulty(student_data)
            
            print(f"\nStudent {i+1} (ID: {student_id}):")
            print(f"- Score: {student_data['score']:.2f}")
            print(f"- Pass Rate: {student_data['pass_rate']:.2f}")
            print(f"- Recommended Difficulty: {prediction['difficulty_level']} (0-4)")
            print(f"  Confidence: {prediction['confidence_scores'][f'level_{prediction['difficulty_level']}']*100:.1f}%")
    
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
