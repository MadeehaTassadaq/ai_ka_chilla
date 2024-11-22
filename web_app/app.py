
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Assuming you have a pre-trained model saved as 'model.h5'
model = load_model('model/risk_prediction_model.h5')  # Load your trained model
# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    error_message = None
    
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
            
            # Use the same scaler that was used during training
            input_data_scaled = scaler.transform(input_data)  
        
            # Make prediction
            predictions = model.predict(input_data_scaled)
            print("Prediction probabilities:", predictions)
            predicted_class = np.argmax(predictions,axis=1)[0]

            # Map prediction to risk levels
            risk_levels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            prediction = risk_levels[predicted_class]

             # Set alert class based on the predicted risk level
            if predicted_class == 0:
                alert_class = "alert-success"  # Green for Low Risk
            elif predicted_class == 1:
                alert_class = "alert-primary"  # Blue for Medium Risk
            elif predicted_class == 2:
                alert_class = "alert-danger"  # Red for High Risk

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, error_message=error_message)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
