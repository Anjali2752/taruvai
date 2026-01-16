import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError as MSE

# -----------------------
# Load data
# -----------------------
df = pd.read_csv("../data/weather_2021_2025.csv")
print(df.columns)
print(df.head())
features = ['temperature', 'humidity', 'rainfall', 'aqi']
data = df[features].values

# -----------------------
# Scale
# -----------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# -----------------------
# Create sequences
# -----------------------
def create_seq(data, seq_len=30):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X, y = create_seq(scaled)

# -----------------------
# LSTM Model
# -----------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 4)),
    LSTM(32),
    Dense(4)
])

# IMPORTANT: use object-based loss & metrics
model.compile(
    optimizer=Adam(),
    loss=MeanSquaredError(),
    metrics=[MSE()]
)

# -----------------------
# Train
# -----------------------
model.fit(
    X, y,
    epochs=20,
    batch_size=32,
    verbose=1
)

# -----------------------
# SAVE (CRITICAL)
# -----------------------
model.save("../models/weather_lstm.keras")  # ✅ SAFE FORMAT
