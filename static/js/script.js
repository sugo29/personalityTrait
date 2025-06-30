document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('personalityForm');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const predictBtn = document.querySelector('.predict-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide previous results
        resultDiv.style.display = 'none';
        errorDiv.style.display = 'none';
        
        // Show loading state
        predictBtn.disabled = true;
        btnText.style.display = 'none';
        loader.style.display = 'block';
        
        try {
            // Get form data
            const formData = new FormData(form);
            
            // Validate inputs
            const inputs = form.querySelectorAll('input[required], select[required]');
            for (let input of inputs) {
                if (!input.value) {
                    const label = input.labels ? input.labels[0].textContent : input.name;
                    throw new Error(`Please fill in the ${label} field.`);
                }
            }
            
            // Send prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Show success result
                displayResult(data);
            } else {
                // Show error
                displayError(data.error || 'An error occurred while making the prediction.');
            }
            
        } catch (error) {
            console.error('Error:', error);
            displayError(error.message || 'An unexpected error occurred. Please try again.');
        } finally {
            // Reset button state
            predictBtn.disabled = false;
            btnText.style.display = 'inline';
            loader.style.display = 'none';
        }
    });
    
    function displayResult(data) {
        const personalityType = document.querySelector('.personality-type');
        const predictionValue = document.querySelector('.prediction-value');
        const description = document.querySelector('.description');
        
        // Update content with enhanced information
        personalityType.textContent = data.personality_type;
        
        // Enhanced prediction display
        let scoreText = `Prediction Score: ${data.prediction_value.toFixed(3)}`;
        if (data.confidence) {
            scoreText += ` (${data.confidence}% confidence)`;
        }
        if (data.strength) {
            scoreText += ` - ${data.strength} ${data.personality_type}`;
        }
        predictionValue.textContent = scoreText;
        
        // Enhanced description
        let fullDescription = data.description;
        if (data.score_explanation) {
            fullDescription += "\n\n" + data.score_explanation;
        }
        if (data.score_meaning) {
            fullDescription += "\n\nHow scoring works: " + data.score_meaning;
        }
        description.textContent = fullDescription;
        
        // Add personality-specific styling
        const resultContent = document.querySelector('.result');
        if (data.personality_type.toLowerCase() === 'introvert') {
            resultContent.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            personalityType.textContent = '🤫 ' + data.personality_type;
        } else {
            resultContent.style.background = 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)';
            personalityType.textContent = '🎊 ' + data.personality_type;
        }
        
        // Show result
        resultDiv.style.display = 'block';
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    function displayError(message) {
        const errorMessage = document.querySelector('.error-message');
        errorMessage.textContent = message;
        errorDiv.style.display = 'block';
        errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    // Add input validation and formatting
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            // Remove any styling for invalid inputs
            this.style.borderColor = '';
            
            // Validate range based on input constraints
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            const value = parseFloat(this.value);
            
            if (this.value && (value < min || value > max)) {
                this.style.borderColor = '#ff6b6b';
                this.title = `Please enter a value between ${min} and ${max}`;
            } else {
                this.style.borderColor = '';
                this.title = '';
            }
        });
        
        // Format decimal inputs
        input.addEventListener('blur', function() {
            if (this.step === '0.1' && this.value) {
                this.value = parseFloat(this.value).toFixed(1);
            }
        });
    });
    
    // Add form reset functionality
    function resetForm() {
        form.reset();
        resultDiv.style.display = 'none';
        errorDiv.style.display = 'none';
        
        // Reset any custom styling
        numberInputs.forEach(input => {
            input.style.borderColor = '';
            input.title = '';
        });
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            if (form.checkValidity()) {
                form.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to reset form
        if (e.key === 'Escape') {
            resetForm();
        }
    });
    
    // Add smooth scrolling for navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            if (this.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });
    
    // Add loading animation for page transitions
    window.addEventListener('beforeunload', function() {
        document.body.style.opacity = '0.8';
    });
});
