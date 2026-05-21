# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
The objective of this experiment is to build a machine learning model that predicts environmental conditions such as temperature, air pollution (PM2.5), and solar radiation energy using sensor data collected over time. The dataset contains time-series data with features such as humidity, pressure, wind speed, illumination, CO₂ levels, and timestamps. Since the outputs are continuous values, this is treated as a regression problem.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the weather dataset using pandas.
2. Preprocess the data by handling missing values and sorting by time.
3. Select features and create lag variables for temperature and PM2.5.
4. Train Random Forest models to predict temperature and PM2.5 and save the models.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: MANIKANDAN K
RegisterNumber: 212224260150
*/
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# LOAD DATA
df = pd.read_csv("weather data.csv")

df.columns = df.columns.str.strip()

# TIME
df['time'] = pd.to_datetime(df['time'])

df = df.sort_values('time').reset_index(drop=True)

# REQUIRED COLUMNS
cols = [
    'tem',
    'pm2_5',
    'tsr',
    'hum',
    'pressure',
    'wind_speed',
    'illumination',
    'co2'
]

# CONVERT TO NUMERIC
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# FILL MISSING VALUES
for col in cols:
    df[col] = df[col].interpolate()
    df[col] = df[col].fillna(method='bfill')
    df[col] = df[col].fillna(method='ffill')

# TIME FEATURES
df['hour'] = df['time'].dt.hour

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# TARGETS
targets = ['tem', 'pm2_5', 'tsr']

# LAG FEATURES
for t in targets:
    df[t + '_lag1'] = df[t].shift(1)

# REMOVE ONLY FIRST NULL ROW
df = df.iloc[1:].reset_index(drop=True)

# FEATURES
features = [
    'hum',
    'pressure',
    'wind_speed',
    'illumination',
    'co2',
    'hour_sin',
    'hour_cos',
    'tem_lag1',
    'pm2_5_lag1',
    'tsr_lag1'
]

print("--- Feature Engineering Summary ---")
print("Processed rows :", len(df))
print("Final feature set :", features)

# SPLIT DATA
split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[features]
X_test = test[features]

models = {}
results = {}

# LABELS
target_info = {
    'tem': ('Temperature', '°C', 'red'),
    'pm2_5': ('Pollution (PM2.5)', 'µg/m³', 'green'),
    'tsr': ('Energy (Solar Radiation)', 'W/m²', 'orange')
}

# TRAIN MODEL
for target in targets:

    y_train = train[target]
    y_test = test[target]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    models[target] = model

    results[target] = {
        'actual': y_test.values,
        'predicted': pred,
        'r2': r2_score(y_test, pred),
        'mae': mean_absolute_error(y_test, pred)
    }

# PLOT
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for i, target in enumerate(targets):

    label, unit, color = target_info[target]

    res = results[target]

    # ACTUAL VS PREDICTED
    axes[i, 0].plot(
        res['actual'][-150:],
        label='Actual',
        color='black',
        alpha=0.5,
        linewidth=2
    )

    axes[i, 0].plot(
        res['predicted'][-150:],
        label='Predicted',
        color=color,
        linestyle='--',
        linewidth=2
    )

    axes[i, 0].set_title(
        f"{label}: Actual vs Predicted\n"
        f"R²: {res['r2']:.3f} | MAE: {res['mae']:.2f}"
    )

    axes[i, 0].set_ylabel(unit)

    axes[i, 0].legend()

    axes[i, 0].grid(True)

    # FEATURE IMPORTANCE
    importance = pd.Series(
        models[target].feature_importances_,
        index=features
    ).sort_values()

    importance.plot(
        kind='barh',
        ax=axes[i, 1],
        color=color,
        alpha=0.7
    )

    axes[i, 1].set_title(f"Key Drivers: {label}")

plt.tight_layout()

plt.show()
```
## Output:
<img width="832" height="972" alt="image" src="https://github.com/user-attachments/assets/ec22b9a9-b9e6-471e-87f4-85a68604de17" />

## Result:
The Random Forest model successfully predicted temperature, PM2.5 pollution, and solar radiation using weather sensor data with good accuracy. The system also generated next-step predictions and visual graphs comparing actual vs predicted values and showing feature importance.
