import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import gdown

# -------------------- FILE NAMES --------------------
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# -------------------- GOOGLE DRIVE IDS --------------------
MODEL_URL = "https://drive.google.com/uc?id=1babW_YIwHrfZS4fpoSjefezpvzfBgwze"
PIPELINE_URL = "https://drive.google.com/uc?id=1WbnvjatqNjyB-Wq2RfpuDzv6GdWSp24C"


# -------------------- LOAD ARTIFACTS --------------------
@st.cache_resource
def load_artifacts():
    """Download (if needed) and load trained model and preprocessing pipeline."""

    if not os.path.exists(MODEL_FILE):
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False, fuzzy=True)

    if not os.path.exists(PIPELINE_FILE):
        gdown.download(PIPELINE_URL, PIPELINE_FILE, quiet=False, fuzzy=True)

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline


def get_feature_schema(pipeline):
    """Return ordered column names expected by the pipeline."""
    num_cols = list(pipeline.transformers_[0][2])
    cat_cols = list(pipeline.transformers_[1][2])
    return num_cols + cat_cols, num_cols, cat_cols


def make_single_row(feature_order, values):
    """Create a single-row DataFrame respecting the pipeline column order."""
    return pd.DataFrame([{col: values.get(col, np.nan) for col in feature_order}])


# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Housing Price Predictor", layout="wide")
st.title("Housing Price Predictor")
st.caption("Streamlit interface for the California housing model.")

# Load model + pipeline
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

# -------------------- SINGLE PREDICTION --------------------
with single_tab:
    st.subheader("Predict a single house")
    cols = st.columns(3)

    user_values = {}
    for idx, col_name in enumerate(num_cols):
        col = cols[idx % len(cols)]
        user_values[col_name] = col.number_input(
            label=col_name.replace("_", " ").title(),
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
        label=cat_col.replace("_", " ").title(),
        options=["<select>", "<NA>"] + sorted(categories),
        index=2 if defaults.get(cat_col) in categories else 0,
        help="Use <NA> for missing category; unknown categories are ignored",
    )

    if user_values[cat_col] in {"<select>", "<NA>"}:
        user_values[cat_col] = np.nan

    if st.button("Predict price", type="primary"):
        single_df = make_single_row(feature_order, user_values)
        transformed = pipeline.transform(single_df)
        pred = model.predict(transformed)[0]
        st.success(f"Estimated median house value: ${pred:,.0f}")

# -------------------- BATCH PREDICTION --------------------
with batch_tab:
    st.subheader("Predict from CSV")
    st.markdown(
        "CSV must contain the following columns: "
        + ", ".join(feature_order)
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        input_df = pd.read_csv(uploaded)
        st.write("Preview", input_df.head())

        if set(feature_order).issubset(input_df.columns):
            if st.button("Run batch prediction", type="primary"):
                ordered = input_df[feature_order]
                preds = model.predict(pipeline.transform(ordered))

                output_df = input_df.copy()
                output_df["median_house_value"] = preds

                csv_buf = io.StringIO()
                output_df.to_csv(csv_buf, index=False)

                st.download_button(
                    label="Download predictions",
                    data=csv_buf.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
                st.success("Predictions ready. Download the file above.")
        else:
            missing = set(feature_order) - set(input_df.columns)
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
