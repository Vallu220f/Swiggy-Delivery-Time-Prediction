"""
app.py — Streamlit UI for the Swiggy delivery-time prediction model.

Run with:
    streamlit run app.py

Expects model.pkl and metadata.json in the same folder (created by train_model.py).
"""

import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Swiggy Delivery Time Predictor", page_icon="🛵", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_metadata():
    with open("metadata.json") as f:
        return json.load(f)


model = load_model()
meta = load_metadata()

st.title("🛵 Swiggy Delivery Time Predictor")
st.caption("Estimate delivery time (in minutes) based on order & rider conditions.")

with st.form("prediction_form"):
    st.subheader("Rider details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Rider age", min_value=int(meta["numerical_ranges"]["age"]["min"]),
                               max_value=int(meta["numerical_ranges"]["age"]["max"]),
                               value=int(meta["numerical_ranges"]["age"]["median"]))
    with col2:
        ratings = st.number_input("Rider rating", min_value=float(meta["numerical_ranges"]["ratings"]["min"]),
                                   max_value=float(meta["numerical_ranges"]["ratings"]["max"]),
                                   value=float(meta["numerical_ranges"]["ratings"]["median"]), step=0.1)

    vehicle_condition = st.selectbox("Vehicle condition", sorted(meta["discrete_options"]["vehicle_condition"]))
    type_of_vehicle = st.selectbox("Type of vehicle", meta["nominal_cols"]["type_of_vehicle"])

    st.subheader("Order details")
    col3, col4 = st.columns(2)
    with col3:
        type_of_order = st.selectbox("Type of order", meta["nominal_cols"]["type_of_order"])
        weather = st.selectbox("Weather", meta["nominal_cols"]["weather"])
        traffic = st.selectbox("Traffic", meta["ordinal_cols"]["traffic"])
        festival = st.selectbox("Festival day?", meta["nominal_cols"]["festival"])
    with col4:
        city_name = st.selectbox("City", meta["nominal_cols"]["city_name"])
        city_type = st.selectbox("City type", meta["ordinal_cols"]["city_type"])
        order_time_of_day = st.selectbox("Time of day", meta["ordinal_cols"]["order_time_of_day"])
        order_day_of_week = st.selectbox("Day of week", meta["nominal_cols"]["order_day_of_week"])

    st.subheader("Timing & logistics")
    col5, col6 = st.columns(2)
    with col5:
        order_day = st.number_input("Order day (1-31)", min_value=1, max_value=31,
                                     value=int(meta["numerical_ranges"]["order_day"]["median"]))
        order_month = st.number_input("Order month (1-12)", min_value=1, max_value=12,
                                       value=int(meta["numerical_ranges"]["order_month"]["median"]))
        is_weekend = st.selectbox("Is weekend?", sorted(meta["discrete_options"]["is_weekend"]))
    with col6:
        order_time_hour = st.number_input("Order hour (0-23)", min_value=0, max_value=23,
                                           value=int(meta["numerical_ranges"]["order_time_hour"]["median"]))
        pickup_time_minutes = st.number_input(
            "Pickup time (minutes)",
            min_value=float(meta["numerical_ranges"]["pickup_time_minutes"]["min"]),
            max_value=float(meta["numerical_ranges"]["pickup_time_minutes"]["max"]),
            value=float(meta["numerical_ranges"]["pickup_time_minutes"]["median"]),
        )
        multiple_deliveries = st.selectbox("Multiple deliveries", sorted(meta["discrete_options"]["multiple_deliveries"]))

    distance = st.number_input(
        "Distance (km)",
        min_value=float(meta["numerical_ranges"]["distance"]["min"]),
        max_value=float(meta["numerical_ranges"]["distance"]["max"]),
        value=float(meta["numerical_ranges"]["distance"]["median"]),
    )

    submitted = st.form_submit_button("Predict delivery time")

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "ratings": ratings,
        "weather": weather,
        "traffic": traffic,
        "vehicle_condition": vehicle_condition,
        "type_of_order": type_of_order,
        "type_of_vehicle": type_of_vehicle,
        "multiple_deliveries": multiple_deliveries,
        "festival": festival,
        "city_type": city_type,
        "city_name": city_name,
        "order_day": order_day,
        "order_month": order_month,
        "order_day_of_week": order_day_of_week,
        "is_weekend": is_weekend,
        "pickup_time_minutes": pickup_time_minutes,
        "order_time_hour": order_time_hour,
        "order_time_of_day": order_time_of_day,
        "distance": distance,
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"### Estimated delivery time: **{prediction:.1f} minutes**")

    with st.expander("See input sent to model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))