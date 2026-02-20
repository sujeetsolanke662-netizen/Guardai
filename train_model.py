# Import necessary libraries
import pandas as pd
from sklearn.ensemble import IsolationForest

# --- Step 1: Load the Log Data ---
try:
    # Load the log data from the CSV file
    df = pd.read_csv('access_log.csv')
    print("✅ Log file loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'access_log.csv' not found. Please ensure the file is in the same directory.")
    exit() # Stop the script if the file isn't found

# --- Step 2: Feature Engineering (Convert Logs to Numbers) ---
# We create numerical features that the model can understand.
print("⚙️  Performing feature engineering...")

# Create a copy to avoid changing the original DataFrame during this step
df_features = df.copy()

# Feature 1: Count how many total requests each IP address has made.
df_features['ip_request_count'] = df_features['IP'].map(df_features['IP'].value_counts())

# Feature 2: Check if a request resulted in a client or server error (status >= 400).
df_features['is_error_status'] = df_features['Status'].apply(lambda status: 1 if status >= 400 else 0)

# Select only the numerical features we created for the model
features_for_model = df_features[['ip_request_count', 'is_error_status']]
print("✅ Features created: 'ip_request_count', 'is_error_status'")

# --- Step 3: Create and Configure the Model ---
# contamination='auto' lets the algorithm decide the anomaly threshold.
# random_state=42 ensures the results are the same every time we run the code.
model = IsolationForest(contamination='auto', random_state=42)
print("🌲 Isolation Forest model created.")

# --- Step 4: Train the Model ---
# The .fit() method teaches the model what "normal" data looks like.
print("🏃 Training the model...")
model.fit(features_for_model)
print("✅ Model training complete.")

# --- Step 5: Get Predictions ---
# The .predict() method flags data points as normal (1) or anomaly (-1).
print("🔍 Detecting anomalies...")
predictions = model.predict(features_for_model)

# Add the prediction results back to our original DataFrame
df['is_anomaly'] = predictions
print("✅ Anomaly detection complete.")

# --- Step 6: Display the Results ---
# Filter the DataFrame to show only the logs that were flagged as suspicious.
suspicious_logs = df[df['is_anomaly'] == -1]

print("\n" + "="*40)
print("🚨 DETECTED SUSPICIOUS LOGS 🚨")
print("="*40)

if suspicious_logs.empty:
    print("👍 No suspicious activities were detected in the logs.")
else:
    # Print the suspicious logs for review
    print(suspicious_logs)

print("="*40)