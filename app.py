from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Define paths for model and scalers
model_path = os.path.join("H:/New folder/car_purchase/models", "car_price_modelui.keras")  # update as needed
scaler_x_path = os.path.join("H:/New folder/car_purchase/models", "scalerui.pkl")         # input scaler
scaler_y_path = os.path.join("H:/New folder/car_purchase/models", "y_scalerui.pkl")         # output scaler

# Load the trained model and scalers
try:
    model = load_model(model_path)
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    app.logger.info("Model and scalers loaded successfully.")
except Exception as e:
    app.logger.error("Error loading model/scalers: %s", e)
    # If model loading fails, you might want to exit or raise an exception.

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/services')
def services():
    return render_template('service.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        try:
            # Collect input values from the form
            data = np.array([[float(request.form['gender']),
                               float(request.form['age']),
                               float(request.form['annual_salary']),
                               float(request.form['credit_score']),
                               float(request.form['net_worth'])]])
            
            # Scale the input data using the trained input scaler
            data_scaled = scaler_x.transform(data)
            
            # Get the scaled prediction from the model
            prediction_scaled = model.predict(data_scaled)[0][0]
            
            # Reverse scale the prediction to get the actual value
            prediction = scaler_y.inverse_transform(np.array([[prediction_scaled]]))[0][0]
            
            # Log the prediction for debugging
            app.logger.info("Prediction: %s", prediction)
            
            return jsonify({'prediction': round(float(prediction), 2)})
        except Exception as e:
            app.logger.error("Error during prediction: %s", e)
            return jsonify({'error': str(e)})
    return render_template('test.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    try:
        save_path = os.path.join("H:/New folder", file.filename)
        file.save(save_path)
        app.logger.info("File uploaded successfully: %s", save_path)
        return jsonify({'message': 'File uploaded successfully'})
    except Exception as e:
        app.logger.error("Error uploading file: %s", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
