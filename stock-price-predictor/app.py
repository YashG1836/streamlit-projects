import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow import random as tf_random
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Basic seeds for reproducibility
np.random.seed(42)
tf_random.set_seed(42)

WINDOW = 60  # lookback window for sequences
DEFAULT_START = "2012-01-01"


def make_sequences(series: np.ndarray, window: int):
    """Build overlapping windowed sequences for LSTM."""
    x, y = [], []
    for i in range(window, len(series)):
        x.append(series[i - window:i])
        y.append(series[i])
    return np.array(x), np.array(y)


def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    true_dir = np.sign(np.diff(y_true, axis=0))
    pred_dir = np.sign(np.diff(y_pred, axis=0))
    return float((true_dir == pred_dir).mean())


def forecast_next_days(model, scaler, full_scaled: np.ndarray, days: int, window: int):
    """Iteratively forecast `days` steps ahead using last known window."""
    history = list(full_scaled[-window:].flatten())
    preds_scaled = []
    for _ in range(days):
        window_arr = np.array(history[-window:]).reshape(1, window, 1)
        next_scaled = model.predict(window_arr, verbose=0)[0][0]
        preds_scaled.append(next_scaled)
        history.append(next_scaled)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()


st.title("On-Demand Stock Trainer & Forecaster")

st.markdown("""
- Enter a ticker; we fetch data up to **today**, train a small LSTM just for that ticker, and forecast.
- Shows train/validation curves, test-set metrics, next-day + 5-day forecast, and a simple buy/hold hint.
- Training runs locally each time you press the button; for speed keep epochs small.
""")

ticker = st.text_input("Ticker", "GOOG").strip().upper()
start_date = st.text_input("Start date (YYYY-MM-DD)", DEFAULT_START)
end_date = datetime.today().strftime("%Y-%m-%d")
st.caption("Using accuracy-first defaults: epochs=80 with early stopping, batch_size=32. Training runs until val_loss stops improving.")
epochs = 80
batch_size = 32

if st.button("Train and Predict"):
    with st.spinner("Downloading data..."):
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
        except Exception as exc:
            st.error(f"Download failed: {exc}")
            st.stop()

    if df.empty or "Close" not in df:
        st.error("No price data found for that ticker and date range.")
        st.stop()

    close_prices = df[["Close"]].dropna()
    if len(close_prices) <= WINDOW + 10:
        st.error("Not enough data to build sequences. Try an earlier start date.")
        st.stop()

    st.write("Fetched rows:", len(close_prices))

    # Train/test split
    train_size = int(len(close_prices) * 0.8)
    train_df = close_prices.iloc[:train_size]
    test_df = close_prices.iloc[train_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    # Build sequences
    x_train, y_train = make_sequences(train_scaled, WINDOW)
    x_test, y_test = make_sequences(np.vstack([train_scaled[-WINDOW:], test_scaled]), WINDOW)

    # Reshape for LSTM
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    model = build_model((WINDOW, 1))

    with st.spinner("Training model..."):
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        ]
        history = model.fit(
            x_train,
            y_train,
            epochs=int(epochs),
            batch_size=int(batch_size),
            validation_split=0.1,
            shuffle=True,
            verbose=0,
            callbacks=callbacks,
        )

    st.subheader("Training curves")
    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(history.history["loss"], label="train")
    ax_loss.plot(history.history["val_loss"], label="val")
    ax_loss.set_ylabel("MSE loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()
    st.pyplot(fig_loss)

    # Evaluate on test
    preds_scaled = model.predict(x_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled)
    actual = scaler.inverse_transform(y_test)

    mae = float(np.mean(np.abs(preds - actual)))
    rmse = float(math.sqrt(np.mean((preds - actual) ** 2)))
    mape = float(np.mean(np.abs((actual - preds) / actual)) * 100)
    dir_acc = directional_accuracy(actual.flatten(), preds.flatten())

    st.subheader("Test metrics")
    st.write({
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE %": round(mape, 2),
        "Directional accuracy": round(dir_acc, 3) if not math.isnan(dir_acc) else "n/a",
    })

    st.subheader("Actual vs Predicted (test set)")
    fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
    ax_pred.plot(actual, label="Actual")
    ax_pred.plot(preds, label="Predicted")
    ax_pred.legend()
    st.pyplot(fig_pred)

    # Next-day and 5-day forecast using full data
    full_scaled = scaler.transform(close_prices)
    next_5 = forecast_next_days(model, scaler, full_scaled, days=5, window=WINDOW)
    next_day = next_5[0]

    last_date = close_prices.index[-1]
    forecast_dates = [(last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(len(next_5))]
    forecast_with_dates = [
        {"date": d, "predicted_close": float(v)} for d, v in zip(forecast_dates, next_5)
    ]

    st.subheader("Forecast")
    st.write({
        "Last close": float(close_prices.iloc[-1, 0]),
        "Next day forecast": {"date": forecast_dates[0], "value": float(next_day)},
        "5-day forecast": forecast_with_dates,
    })

    # Simple buy/hold rule
    last_close = float(close_prices.iloc[-1, 0])
    avg_future = float(np.mean(next_5))
    change_pct = ((avg_future - last_close) / last_close) * 100
    if change_pct > 1.0:
        decision = "Buy"
    elif change_pct < -1.0:
        decision = "Avoid / Sell"
    else:
        decision = "Hold / Neutral"

    st.subheader("Decision hint")
    st.write({
        "Avg 5-day expected change %": round(change_pct, 2),
        "Suggestion": decision,
    })
else:
    st.info("Set ticker/dates, adjust epochs if needed, then press 'Train and Predict'.")
