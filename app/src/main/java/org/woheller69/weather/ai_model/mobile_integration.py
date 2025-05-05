import numpy as np
import onnxruntime as ort
import joblib
import os

def load_model(model_path, scaler_path):
    """Load ONNX model and scaler for mobile use."""
    session = ort.InferenceSession(model_path)
    scaler = joblib.load(scaler_path)
    return session, scaler

def predict_rainfall(input_data, session, scaler, input_name):
    """Run prediction on mobile device."""
    # Preprocess input
    if isinstance(input_data, list):
        input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    
    # Run inference
    outputs = session.run(None, {input_name: input_data_scaled})
    return outputs[0][0]

# Mobile integration entry point
def mobile_predict(input_data, model_path='rf_model.onnx', scaler_path='scaler.pkl'):
    """API for mobile app to call."""
    try:
        # Load model and scaler
        session, scaler = load_model(model_path, scaler_path)
        input_name = session.get_inputs()[0].name
        
        # Predict
        prediction = predict_rainfall(input_data, session, scaler, input_name)
        return float(prediction)
    except Exception as e:
        return f"Error: {str(e)}"

# Example for testing
if __name__ == "__main__":
    # Ensure model and scaler files are in the app's assets or storage
    sample_input = [30.0, 24.0, 80, 1, 350]
    result = mobile_predict(sample_input)
    print(f"Mobile Prediction: {result:.2f} mm")