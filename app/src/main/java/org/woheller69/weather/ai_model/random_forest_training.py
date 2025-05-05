# Importing libraries for Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Loading and preprocessing the dataset (assuming it's already uploaded)
data = pd.read_csv('Science Garden Daily Data.csv')

# Handling missing values (-1 in RAINFALL)
data['RAINFALL'] = data['RAINFALL'].replace(-1, 0)

# Selecting features and target
features = ['TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION']
target = 'RAINFALL'

# Creating feature and target arrays
X = data[features].values
y = data[target].values

# Splitting into training and testing sets (80-20 split)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Making predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Random Forest - Mean Squared Error: {mse:.2f}")
print(f"Random Forest - Mean Absolute Error: {mae:.2f}")

# Plotting actual vs predicted rainfall
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Rainfall', color='blue')
plt.plot(y_pred, label='Predicted Rainfall', color='orange')
plt.title('Random Forest: Actual vs Predicted Rainfall')
plt.xlabel('Test Sample Index')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)
plt.savefig('rf_predictions.png')
plt.show()

# Saving the model and scaler for mobile integration
import joblib
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')