import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def simulate_sensor_data(machine_id, start_date, days, freq='5min'):
    timestamps = pd.date_range(start=start_date, periods=days * 24 * (60 // int(freq[:-3])), freq=freq)
    data = pd.DataFrame({
        'timestamp': timestamps,
        'machine_id': machine_id,
        'temperature': np.random.normal(75, 5, len(timestamps)),
        'vibration': np.random.normal(0.5, 0.1, len(timestamps)),
        'pressure': np.random.normal(30, 3, len(timestamps)),
    })

    # Introduire des anomalies (simulateur de pannes)
    if np.random.rand() < 0.3:  # 30% des machines ont une panne simulée
        anomaly_idx = np.random.choice(len(data), size=10, replace=False)
        data.loc[anomaly_idx, 'vibration'] += np.random.normal(1.0, 0.2, 10)
        data['failure'] = 0
        data.loc[anomaly_idx, 'failure'] = 1
    else:
        data['failure'] = 0

    return data

if __name__ == "__main__":
    all_data = pd.concat([
        simulate_sensor_data(f"MACHINE_{i}", datetime.now() - timedelta(days=30), 30)
        for i in range(5)
    ])
    
    os.makedirs("data/raw", exist_ok=True)
    all_data.to_csv("data/raw/simulated_sensor_data.csv", index=False)
    print("✅ Données simulées enregistrées dans data/raw/")