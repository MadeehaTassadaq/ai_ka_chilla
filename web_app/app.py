from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model/risk_prediction_model.h5')

# Initialize Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None  # Default prediction value
    error_message = None  # For displaying errors, if any

    if request.method == 'POST':
        try:
            # Extract data from the form
            age = float(request.form['age'])
            systolic_bp = float(request.form['systolic_bp'])
            diastolic_bp = float(request.form['diastolic_bp'])
            bsr = float(request.form['bsr'])
            body_temp = float(request.form['body_temp'])
            heart_rate = float(request.form['heart_rate'])

            # Prepare input data for the model
            input_data = np.array([[age, systolic_bp, diastolic_bp, bsr, body_temp, heart_rate]])

            # Make prediction
            predictions = model.predict(input_data)
            predicted_class = np.argmax(predictions, axis=1)[0]

            # Map prediction to risk levels
            risk_levels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            prediction = risk_levels[predicted_class]

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, error_message=error_message)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
