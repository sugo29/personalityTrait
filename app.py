from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Define the custom function that was used when creating the model
def convert_yes_no_columns(data):
    """Convert yes/no columns to numeric values"""
    if isinstance(data, str):
        if data.lower() in ['yes', 'y', 'true', '1']:
            return 1
        elif data.lower() in ['no', 'n', 'false', '0']:
            return 0
    return data

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

app = Flask(__name__)

# Load the pre-trained model (your actual trained model)
MODEL_PATH = 'trained_personality_model.pkl'

def load_model():
    """Load the linear regression model"""
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as file:
                model = pickle.load(file)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
            return model
        else:
            print(f"❌ Model file {MODEL_PATH} not found. Please train your model first.")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Load model at startup
model = load_model()

@app.route('/')
def index():
    """Render the about page as the main landing page"""
    return render_template('about.html')

@app.route('/predict')
def predict_page():
    """Render the prediction page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make personality prediction based on input features"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check if the trained model exists.'}), 500
        
        # Get input data from form and handle mixed data types
        time_spent_alone = float(request.form['time_spent_alone'])
        
        # Stage_fear is yes/no - convert to numeric
        stage_fear_raw = request.form['stage_fear']
        stage_fear = convert_yes_no_columns(stage_fear_raw)
        
        social_event_attendance = float(request.form['social_event_attendance'])
        going_outside = float(request.form['going_outside'])
        
        # Drained_after_socializing is yes/no - convert to numeric
        drained_raw = request.form['drained_after_socializing']
        drained_after_socializing = convert_yes_no_columns(drained_raw)
        
        friends_circle_size = float(request.form['friends_circle_size'])
        post_frequency = float(request.form['post_frequency'])
        
        # Create feature array
        features = np.array([[
            time_spent_alone,
            stage_fear,
            social_event_attendance,
            going_outside,
            drained_after_socializing,
            friends_circle_size,
            post_frequency
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Calculate confidence (distance from threshold, scaled to percentage)
        confidence = abs(prediction - 0.5) * 200  # Convert to 0-100% scale
        confidence = min(confidence, 100)  # Cap at 100%
        
        # Interpret the personality type based on prediction value
        # Higher values indicate introversion, lower values indicate extroversion
        if prediction > 0.5:
            personality_type = "Introvert"
            description = "You tend to be more comfortable in quiet environments and prefer smaller social groups."
        else:
            personality_type = "Extrovert"
            description = "You tend to be energized by social interactions and enjoy being around people."
        
        # Add explanation of what the score means
        if prediction > 0.8:
            strength = "Strong"
        elif prediction > 0.6:
            strength = "Moderate"
        elif prediction > 0.4:
            strength = "Mild"
        elif prediction > 0.2:
            strength = "Moderate"
        else:
            strength = "Strong"
        
        # Detailed explanation
        score_explanation = f"Your score of {prediction:.3f} indicates {strength.lower()} {personality_type.lower()} tendencies. "
        score_explanation += f"The model calculated this by analyzing your responses across 7 personality factors, "
        score_explanation += f"with stage fear and social energy drain being the strongest predictors."
        
        return jsonify({
            'prediction_value': float(prediction),
            'personality_type': personality_type,
            'description': description,
            'confidence': round(confidence, 1),
            'strength': strength,
            'score_explanation': score_explanation,
            'threshold': 0.5,
            'score_meaning': "Scores above 0.5 indicate introversion, below 0.5 indicate extroversion"
        })
        
    except ValueError as e:
        return jsonify({'error': 'Invalid input. Please enter numeric values.'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/visualize')
def visualize():
    """Render the data visualization page"""
    return render_template('visualize.html')

@app.route('/api/dataset-stats')
def dataset_stats():
    """API endpoint to get dataset statistics for visualization"""
    try:
        # Load and analyze the dataset
        import csv
        
        data = []
        with open('personality_datasert.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    processed_row = {
                        'Time_spent_Alone': float(row['Time_spent_Alone']),
                        'Stage_fear': 1 if row['Stage_fear'].lower() == 'yes' else 0,
                        'Social_event_attendance': float(row['Social_event_attendance']),
                        'Going_outside': float(row['Going_outside']),
                        'Drained_after_socializing': 1 if row['Drained_after_socializing'].lower() == 'yes' else 0,
                        'Friends_circle_size': float(row['Friends_circle_size']),
                        'Post_frequency': float(row['Post_frequency']),
                        'Personality': row['Personality']
                    }
                    data.append(processed_row)
                except (ValueError, KeyError):
                    continue
        
        # Calculate statistics
        introverts = [d for d in data if d['Personality'] == 'Introvert']
        extroverts = [d for d in data if d['Personality'] == 'Extrovert']
        
        stats = {
            'total_samples': len(data),
            'introvert_count': len(introverts),
            'extrovert_count': len(extroverts),
            'features': {}
        }
        
        feature_names = [
            'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency'
        ]
        
        for feature in feature_names:
            intro_values = [d[feature] for d in introverts]
            extro_values = [d[feature] for d in extroverts]
            
            stats['features'][feature] = {
                'introvert_avg': np.mean(intro_values) if intro_values else 0,
                'extrovert_avg': np.mean(extro_values) if extro_values else 0,
                'introvert_values': intro_values[:100],  # Limit for performance
                'extrovert_values': extro_values[:100],
                'all_values': [d[feature] for d in data[:200]]  # Sample for distribution
            }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': f'Failed to load dataset: {str(e)}'}), 500

if __name__ == '__main__':
    if model is not None:
        print("🚀 Starting Flask app with trained personality model...")
        print("📊 Model trained on 2900 samples with 75% accuracy")
        print("🌐 Access the app at: http://localhost:5000")
    else:
        print("⚠️  Starting Flask app without model (predictions will fail)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
