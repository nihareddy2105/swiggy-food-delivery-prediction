
import gzip
import joblib
import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="wide"
)

@st.cache_resource
def load_model():
    with gzip.open("swiggy_delivery_model.pkl.gz", "rb") as f:
        return joblib.load(f)

st.title("🍔 Food Delivery Time Prediction")
st.caption("Machine learning demonstration for a Swiggy-like food delivery business")

try:
    model = load_model()
except Exception as e:
    st.error("Could not load the model. Check that swiggy_delivery_model.pkl.gz is in the repository.")
    st.stop()

st.info(
    "Enter delivery details to estimate the delivery duration. "
    "This is an academic demonstration using public food-delivery data, "
    "not an official Swiggy system."
)

# Use the model's saved preprocessing pipeline to identify input columns.
preprocessor = model.named_steps["preprocessor"]
numeric_features = list(preprocessor.transformers_[0][2])
categorical_features = list(preprocessor.transformers_[1][2])

st.subheader("Delivery details")

with st.form("prediction_form"):
    inputs = {}

    left, right = st.columns(2)

    with left:
        for col in numeric_features:
            if col in [
                "Delivery_person_Age",
                "Delivery_person_Ratings",
                "Restaurant_latitude",
                "Restaurant_longitude",
                "Delivery_location_latitude",
                "Delivery_location_longitude",
                "multiple_deliveries",
            ]:
                inputs[col] = st.number_input(
                    col.replace("_", " "),
                    value=0.0
                )

    with right:
        for col in categorical_features:
            if col == "Weatherconditions":
                inputs[col] = st.selectbox(
                    col.replace("_", " "),
                    ["Sunny", "Cloudy", "Stormy", "Sandstorms", "Windy", "Fog"]
                )
            elif col == "Road_traffic_density":
                inputs[col] = st.selectbox(
                    col.replace("_", " "),
                    ["Low", "Medium", "High", "Jam"]
                )
            elif col == "Vehicle_condition":
                inputs[col] = st.selectbox(
                    col.replace("_", " "),
                    ["0", "1", "2", "3"]
                )
            elif col == "Type_of_order":
                inputs[col] = st.selectbox(
                    col.replace("_", " "),
                    ["Snack", "Meal", "Drinks", "Buffet"]
                )
            elif col == "Type_of_vehicle":
                inputs[col] = st.selectbox(
                    col.replace("_", " "),
                    ["motorcycle", "scooter", "electric_scooter", "bicycle"]
                )
            elif col == "Festival":
                inputs[col] = st.selectbox(
                    col.replace("_", " "),
                    ["No", "Yes"]
                )
            elif col == "City":
                inputs[col] = st.selectbox(
                    col.replace("_", " "),
                    ["Metropolitian", "Urban", "Semi-Urban"]
                )
            elif col == "Time_Orderd":
                inputs[col] = "10:30"
            elif col == "Time_Order_picked":
                inputs[col] = "10:40"
            elif col == "Order_Date":
                inputs[col] = "15-03-2022"
            else:
                inputs[col] = st.text_input(
                    col.replace("_", " "),
                    value="Unknown"
                )

    submitted = st.form_submit_button("Predict delivery time")

if submitted:
    input_df = pd.DataFrame([inputs])

    # Match the model's expected feature order
    expected_features = numeric_features + categorical_features
    input_df = input_df.reindex(columns=expected_features)

    try:
        prediction = float(model.predict(input_df)[0])
        st.success(f"Estimated delivery time: {prediction:.1f} minutes")
        st.caption(
            "This is a model estimate, not a guaranteed delivery time."
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.subheader("Model validation results")

c1, c2, c3 = st.columns(3)
c1.metric("Validation MAE", "3.85 min")
c2.metric("Validation RMSE", "4.92 min")
c3.metric("Baseline MAE", "7.58 min")

st.caption(
    "Metrics are from the validation split used during model development. "
    "Results on future orders may differ."
)
