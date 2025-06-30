# Personality Trait Predictor

A Flask web application that predicts whether someone is an introvert or extrovert based on behavioral patterns using a linear regression model.

## Features

- **Interactive Web Interface**: Modern, responsive design with smooth animations
- **7 Key Personality Factors**: 
  - Time Spent Alone
  - Stage Fear
  - Social Event Attendance
  - Going Outside Frequency
  - Energy Drain After Socializing
  - Friends Circle Size
  - Social Media Post Frequency
- **Real-time Prediction**: Instant personality type prediction using machine learning
- **Mobile Responsive**: Works perfectly on all device sizes
- **Beautiful UI**: Modern gradient design with smooth transitions

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd personalityTrait
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Fill in the personality assessment form** with your behavioral patterns

4. **Click "Predict My Personality"** to get your result

## Model Requirements

Your `linear_regression_model.pkl` should be a scikit-learn LinearRegression model that:
- Takes 7 numerical features as input
- Returns a prediction between 0 (Introvert) and 1 (Extrovert)
- Is saved using pickle

Example of creating and saving a compatible model:
```python
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

# Your training data
X = np.array([...])  # Shape: (n_samples, 7)
y = np.array([...])  # Shape: (n_samples,) - 0 for Introvert, 1 for Extrovert

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```
### User Experience
- Smooth animations and transitions
- Loading states for predictions
- Clear result presentation
- Keyboard shortcuts (Ctrl+Enter to submit, Escape to reset)

## Customization

### Styling
Modify `static/css/style.css` to change the appearance:
- Color schemes are defined using CSS variables
- Responsive breakpoints can be adjusted
- Animation timing and effects can be customized

### Prediction Logic
Modify the prediction interpretation in `app.py`:
- Adjust the threshold for introvert/extrovert classification
- Add more detailed personality descriptions
- Include confidence scores

![image](https://github.com/user-attachments/assets/c949ba17-e716-481d-a88b-affbfb313ee4)

![image](https://github.com/user-attachments/assets/8c36d8d7-57ff-4295-8ce7-574f35fedeec)

![image](https://github.com/user-attachments/assets/8bab72ca-b978-4c0b-a69f-7e4defca185a)

![image](https://github.com/user-attachments/assets/c0baaf8b-ad17-4acc-b131-0601054c2610)



