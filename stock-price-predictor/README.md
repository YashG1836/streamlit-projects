# Stock Price Prediction (Predict_Stock_Price_ML)

## Overview
This project experiments with predicting stock prices using time-series and machine learning techniques. It contains example notebooks and a Streamlit `app.py` for visualizing predictions.

## How it works (high level)
- Data collection: historical price series (open/high/low/close/volume).
- Preprocessing: resampling, handling missing values, scaling (MinMax or StandardScaler), and creating supervised windows of past timesteps as features.
- Model approaches:
  - Classical/regression: linear regression, random forest on handcrafted features (lags, moving averages).
  - Deep learning: LSTM or GRU networks trained on sequences of past prices to predict next-step or multi-step forecasts.
- Evaluation: use walk-forward validation and metrics like RMSE and MAPE.

## Typical pipeline details
- Normalize inputs before training and inverse-transform predictions.
- Use sliding windows (e.g., 60 timesteps) as model inputs for LSTM.
- Split data by time (train/validation/test) to avoid leakage.

## Files
- `app.py` and `app_temp.py`: Streamlit apps to visualize data and model forecasts.
- Notebooks: exploratory workflows and model training experiments.

## Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn, tensorflow or pytorch, streamlit

Install with:

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

## Notes & improvements
- Time-series forecasting is sensitive to lookahead bias; prefer walk-forward validation.
- For production, consider probabilistic forecasts and prediction intervals rather than point estimates.
