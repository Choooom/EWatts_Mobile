import numpy as np
import onnxruntime as ort
import joblib

class RainfallPredictor:
    def __init__(self, model_path='rf_model.onnx', scaler_path='scaler.pkl'):
        # Load the ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Load the scaler
        self.scaler = joblib.load(scaler_path)

    def preprocess(self, input_data):
        # Ensure input_data is a numpy array with shape (1, 5)
        if isinstance(input_data, list):
            input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)
        elif isinstance(input_data, np.ndarray):
            input_data = input_data.astype(np.float32).reshape(1, -1)
        
        # Scale the input data
        input_data_scaled = self.scaler.transform(input_data)
        return input_data_scaled

    def predict(self, input_data):
        # Preprocess the input
        input_data_scaled = self.preprocess(input_data)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data_scaled})
        return outputs[0][0]  # Return the predicted rainfall

# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = RainfallPredictor()
    
    # Example input: [TMAX, TMIN, RH, WIND_SPEED, WIND_DIRECTION]
    sample_input = [30.0, 24.0, 80, 1, 350]
    
    # Predict rainfall
    prediction = predictor.predict(sample_input)
    print(f"Predicted Rainfall: {prediction:.2f} mm")