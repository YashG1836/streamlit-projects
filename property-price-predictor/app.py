import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import gdown

# ==================== FILE NAMES ====================
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# ==================== GOOGLE DRIVE FILES ====================
MODEL_URL = "https://drive.google.com/uc?id=1babW_YIwHrfZS4fpoSjefezpvzfBgwze"
PIPELINE_URL = "https://drive.google.com/uc?id=1WbnvjatqNjyB-Wq2RfpuDzv6GdWSp24C"


# ==================== LOAD MODEL & PIPELINE ====================
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_FILE):
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False, fuzzy=True)

    if not os.path.exists(PIPELINE_FILE):
        gdown.download(PIPELINE_URL, PIPELINE_FILE, quiet=False, fuzzy=True)

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline


# ==================== HELPERS ====================
def get_feature_schema(pipeline):
    num_cols = list(pipeline.transformers_[0][2])
    cat_cols = list(pipeline.transformers_[1][2])
    return num_cols + cat_cols, num_cols, cat_cols


def make_single_row(feature_order, values):
    return pd.DataFrame([{col: values.get(col, np.nan) for col in feature_order}])


def enforce_dtypes(df, num_cols, cat_cols):
    """CRITICAL FIX: ensure sklearn-compatible dtypes."""
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        df[c] = df[c].astype("object")
    return df


# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="Housing Price Predictor", layout="wide")
st.title("Housing Price Predictor")
st.caption("California Housing Price Prediction (Streamlit + ML)")

with st.spinner("Loading model and pipeline..."):
    model, pipeline = load_artifacts()

feature_order, num_cols, cat_cols = get_feature_schema(pipeline)

# Defaults
defaults = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 30,
    "total_rooms": 2000,
    "total_bedrooms": 400,
    "population": 900,
    "households": 350,
    "median_income": 4.0,
    "ocean_proximity": "NEAR BAY",
}

single_tab, batch_tab = st.tabs(["Single Prediction", "Batch Prediction"])

# ==================== SINGLE PREDICTION ====================
with single_tab:
    st.subheader("Predict a single house")
    cols = st.columns(3)

    user_values = {}
    for idx, col_name in enumerate(num_cols):
        col = cols[idx % 3]
        user_values[col_name] = col.number_input(
            col_name.replace("_", " ").title(),
            value=float(defaults.get(col_name, 0.0)),
            step=0.1,
            format="%.3f",
        )

    cat_col = cat_cols[0]
    categories = list(
        pipeline.transformers_[1][1]
        .named_steps["OneHot"]
        .categories_[0]
    )

    user_values[cat_col] = st.selectbox(
        cat_col.replace("_", " ").title(),
        options=["<select>", "<NA>"] + sorted(categories),
        index=2 if defaults.get(cat_col) in categories else 0,
    )

    if user_values[cat_col] in {"<select>", "<NA>"}:
        user_values[cat_col] = np.nan

    if st.button("Predict price", type="primary"):
        single_df = make_single_row(feature_order, user_values)
        single_df = enforce_dtypes(single_df, num_cols, cat_cols)

        transformed = pipeline.transform(single_df)
        pred = model.predict(transformed)[0]

        st.success(f"Estimated median house value: **${pred:,.0f}**")


# ==================== BATCH PREDICTION ====================
with batch_tab:
    st.subheader("Predict from CSV")
    st.markdown("CSV must contain columns: " + ", ".join(feature_order))

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        input_df = pd.read_csv(uploaded)
        st.write("Preview", input_df.head())

        if set(feature_order).issubset(input_df.columns):
            if st.button("Run batch prediction", type="primary"):
                ordered = input_df[feature_order]
                ordered = enforce_dtypes(ordered, num_cols, cat_cols)

                preds = model.predict(pipeline.transform(ordered))
                output_df = input_df.copy()
                output_df["median_house_value"] = preds

                buffer = io.StringIO()
                output_df.to_csv(buffer, index=False)

                st.download_button(
                    "Download predictions",
                    data=buffer.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

                st.success("Predictions generated successfully!")
        else:
            missing = set(feature_order) - set(input_df.columns)
            st.error(f"Missing columns: {', '.join(missing)}")
