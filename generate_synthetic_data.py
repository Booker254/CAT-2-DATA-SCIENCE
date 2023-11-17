import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic sensor data
np.random.seed(42)

# Generate timestamps for 1 month at 1-hour intervals
timestamps = pd.date_range(start="2023-01-01", end="2023-01-31", freq="H")

# Create a DataFrame for sensor data
sensor_data = pd.DataFrame({
    'Timestamp': timestamps,
    'Vibration': np.random.normal(loc=0, scale=1, size=len(timestamps)),
    'Temperature': np.random.normal(loc=25, scale=5, size=len(timestamps)),
    'Pressure': np.random.normal(loc=100, scale=10, size=len(timestamps)),
    'UsageHours': np.cumsum(np.random.randint(1, 10, size=len(timestamps))),
})

# Simulate a sudden increase in Vibration as a precursor to failure
sensor_data.loc[sensor_data.index > len(sensor_data) * 0.8, 'Vibration'] += 5

# Generate synthetic historical maintenance records
maintenance_timestamps = pd.date_range(start="2023-01-01", end="2023-01-31", freq="24H")  # Maintenance every 24 hours
maintenance_records = pd.DataFrame({
    'Timestamp': maintenance_timestamps,
    'MaintenanceType': np.random.choice(['Routine', 'Corrective'], size=len(maintenance_timestamps)),
    'Details': np.random.choice(['Lubrication', 'Bearing Replacement', 'Calibration'], size=len(maintenance_timestamps)),
})

# Introduce some corrective maintenance events
maintenance_records.loc[maintenance_records['MaintenanceType'] == 'Corrective', 'Details'] = 'Equipment Repair'

# Merge sensor data with maintenance records
merged_data = pd.merge(sensor_data, maintenance_records, on='Timestamp', how='left')

# Introduce missing data to simulate real-world scenarios
merged_data.loc[np.random.rand(len(merged_data)) < 0.05, 'Vibration'] = np.nan
merged_data.loc[np.random.rand(len(merged_data)) < 0.05, 'Temperature'] = np.nan
merged_data.loc[np.random.rand(len(merged_data)) < 0.05, 'Pressure'] = np.nan

# Save the generated data to CSV files
sensor_data.to_csv('sensor_data.csv', index=False)
maintenance_records.to_csv('maintenance_records.csv', index=False)
merged_data.to_csv('merged_data.csv', index=False)
