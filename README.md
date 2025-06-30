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

3. **Add your trained model**:
   - Place your `linear_regression_model.pkl` file in the root directory
   - The model should be trained to predict personality type (0 for Introvert, 1 for Extrovert)
   - Expected input features (in order):
     1. Time Spent Alone (hours/day)
     2. Stage Fear (1-10 scale)
     3. Social Event Attendance (events/month)
     4. Going Outside (days/week)
     5. Drained After Socializing (1-10 scale)
     6. Friends Circle Size (number)
     7. Post Frequency (posts/week)

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Fill in the personality assessment form** with your behavioral patterns

4. **Click "Predict My Personality"** to get your result

## Project Structure

```
personalityTrait/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── linear_regression_model.pkl  # Your trained model (add this file)
├── templates/
│   ├── index.html        # Main prediction page
│   └── about.html        # About page
└── static/
    ├── css/
    │   └── style.css     # Styling
    └── js/
        └── script.js     # Frontend JavaScript
```

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

## Features Details

### Input Validation
- All inputs are validated for proper ranges
- Real-time feedback for invalid entries
- Smooth error handling and display

### Responsive Design
- Mobile-first responsive design
- Touch-friendly interface
- Optimized for all screen sizes

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

## Troubleshooting

### Model Not Found
If you see "Model not loaded" error:
1. Ensure `linear_regression_model.pkl` is in the root directory
2. Check that the model was saved using pickle
3. Verify the model is a valid scikit-learn object

### Prediction Errors
If predictions fail:
1. Check that your model expects 7 features in the correct order
2. Ensure input validation ranges match your model's training data
3. Verify the model returns numerical predictions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.
