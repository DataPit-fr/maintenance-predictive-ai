import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = "data/raw/simulated_sensor_data.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

def load_data(path):
    return pd.read_csv(path, parse_dates=['timestamp'])

def remove_outliers(df):
    for col in ['temperature', 'vibration', 'pressure']:
        q_low = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        df = df[(df[col] >= q_low) & (df[col] <= q_high)]
    return df

def feature_engineering(df):
    df = df.sort_values(by=['machine_id', 'timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    # Moyennes glissantes (window=6 pour ~30 min avec un pas de 5 min)
    df['temp_mean_30min'] = df.groupby('machine_id')['temperature'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    df['vib_mean_30min'] = df.groupby('machine_id')['vibration'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    df['pres_mean_30min'] = df.groupby('machine_id')['pressure'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())

    return df

def preprocess():
    df = load_data(RAW_DATA_PATH)
    print(f"✅ Données chargées : {df.shape[0]} lignes")

    df = remove_outliers(df)
    print(f"🔍 Après nettoyage : {df.shape[0]} lignes")

    df = feature_engineering(df)
    print(f"🛠 Caractéristiques créées")

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"💾 Données enregistrées : {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess()