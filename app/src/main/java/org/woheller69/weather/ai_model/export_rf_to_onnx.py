import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the trained model and scaler
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define input type for ONNX (5 features: TMAX, TMIN, RH, WIND_SPEED, WIND_DIRECTION)
initial_type = [('float_input', FloatTensorType([None, 5]))]

# Convert the Random Forest model to ONNX
onnx_model = convert_sklearn(rf_model, initial_types=initial_type)

# Save the ONNX model
with open('rf_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# Save the scaler for preprocessing (optional, for reference)
joblib.dump(scaler, 'scaler.pkl')