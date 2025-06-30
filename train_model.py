"""
Train a linear regression model on the actual personality dataset
"""

import numpy as np
import pickle

def convert_yes_no_to_numeric(value):
    """Convert Yes/No to 1/0"""
    if isinstance(value, str):
        if value.strip().lower() == 'yes':
            return 1
        elif value.strip().lower() == 'no':
            return 0
    return value

def convert_personality_to_numeric(value):
    """Convert Introvert/Extrovert to 1/0"""
    if isinstance(value, str):
        if value.strip().lower() == 'introvert':
            return 1
        elif value.strip().lower() == 'extrovert':
            return 0
    return value

def load_csv_data(filename):
    """Load and process the CSV data"""
    print(f"Loading data from {filename}...")
    
    # Read the CSV file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split(',')
    print(f"Columns: {header}")
    
    # Parse data
    data = []
    for line in lines[1:]:
        row = line.strip().split(',')
        if len(row) == len(header):
            data.append(row)
    
    print(f"Loaded {len(data)} rows")
    
    # Convert to numpy array and process
    features = []
    targets = []
    
    for row in data:
        try:
            # Process features
            time_alone = float(row[0])
            stage_fear = convert_yes_no_to_numeric(row[1])
            social_events = float(row[2])
            going_outside = float(row[3])
            drained = convert_yes_no_to_numeric(row[4])
            friends = float(row[5])
            posts = float(row[6])
            
            # Process target
            personality = convert_personality_to_numeric(row[7])
            
            if all(x is not None for x in [time_alone, stage_fear, social_events, going_outside, drained, friends, posts, personality]):
                features.append([time_alone, stage_fear, social_events, going_outside, drained, friends, posts])
                targets.append(personality)
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid row: {row} - {e}")
            continue
    
    X = np.array(features)
    y = np.array(targets)
    
    print(f"Processed {len(X)} valid samples")
    print(f"Feature shape: {X.shape}")
    print(f"Target distribution - Introvert: {np.sum(y)}, Extrovert: {len(y) - np.sum(y)}")
    
    return X, y

class LinearRegression:
    """Simple linear regression implementation"""
    def __init__(self):
        self.weights = None
        self.bias = None
        self.fitted = False
    
    def fit(self, X, y):
        """Train the model using normal equation"""
        # Add bias term to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: theta = (X^T * X)^(-1) * X^T * y
        try:
            theta = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
            self.bias = theta[0]
            self.weights = theta[1:]
            self.fitted = True
            print("Model training completed successfully!")
        except np.linalg.LinAlgError as e:
            print(f"Training failed: {e}")
            return False
        
        return True
    
    def predict(self, X):
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

def main():
    print("🧠 Training Linear Regression on Personality Dataset")
    print("=" * 60)
    
    # Load the data
    try:
        X, y = load_csv_data('personality_datasert.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if len(X) == 0:
        print("No valid data found!")
        return
    
    # Split data (80% train, 20% test)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train the model
    model = LinearRegression()
    if not model.fit(X_train, y_train):
        print("Training failed!")
        return
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    # Show feature importance (weights)
    feature_names = [
        "Time_spent_Alone",
        "Stage_fear", 
        "Social_event_attendance",
        "Going_outside",
        "Drained_after_socializing",
        "Friends_circle_size",
        "Post_frequency"
    ]
    
    print(f"\nFeature Weights:")
    for name, weight in zip(feature_names, model.weights):
        print(f"  {name}: {weight:.4f}")
    print(f"  Bias: {model.bias:.4f}")
    
    # Test with sample data
    print(f"\nTesting with sample data:")
    test_cases = [
        [15, 1, 2, 1, 1, 5, 2],   # Likely introvert
        [5, 0, 10, 6, 0, 25, 15], # Likely extrovert
    ]
    
    for i, test_data in enumerate(test_cases):
        prediction = model.predict(np.array(test_data))[0]
        personality = "Introvert" if prediction > 0.5 else "Extrovert"
        print(f"  Test {i+1}: {test_data} -> {prediction:.3f} ({personality})")
    
    # Save the model
    model_filename = 'trained_personality_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ Model saved as '{model_filename}'")
    print("You can now update your Flask app to use this trained model!")

if __name__ == "__main__":
    main()
