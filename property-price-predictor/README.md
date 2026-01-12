# Project Gurgaon (Project_gurgaon)

## Overview
This project appears to focus on a regional dataset (housing or local dataset) and contains scripts (`main.py`, `main_old.py`) and a Streamlit `app.py` for exploration and predictions.

## How it works (high level)
- Data ingestion and cleaning: load CSV datasets, handle missing values, and encode categorical variables.
- Feature engineering: create meaningful features (location, area, number of rooms, age, amenities) and scale numeric features.
- Models: regression models such as Linear Regression, Decision Trees, Random Forest or Gradient Boosting to predict prices or target variables.
- Evaluation: use train/test split and metrics like RMSE, MAE, and R².

## Files
- `app.py`: Streamlit app for interactive exploration and simple predictions.
- `main.py` / `main_old.py`: scripts for data processing, model training or experimentation.

## Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn, streamlit

Install with:

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

## Notes & improvements
- Improve feature engineering using geospatial features (distance to landmarks) and better handling of categorical variables.
- Add cross-validation and hyperparameter tuning for robust models.
