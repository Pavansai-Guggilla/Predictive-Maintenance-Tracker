import streamlit as st
import joblib
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load encoders, metadata, and model
try:
    oe = joblib.load("ordinal_encoder.joblib")
    with open("target_classes.json", "r") as f:
        categories = json.load(f)
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    if metadata["model_type"] == "sklearn":
        model = joblib.load("best_model.joblib")
    else:
        model = load_model("best_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Machine Predictive Maintenance Classification")
st.write("Enter the machine parameters below to predict the failure type.")

col1, col2 = st.columns(2)

with col1:
    selected_type = st.selectbox("Select Type", options=oe.categories_[0])

with col2:
    air_temp = st.number_input("Air temperature [K]", value=300.0)

with col1:
    process_temp = st.number_input("Process temperature [K]", value=310.0)

with col2:
    rotational_speed = st.number_input("Rotational speed [rpm]", value=1500.0)

with col1:
    torque = st.number_input("Torque [Nm]", value=40.0)

with col2:
    tool_wear = st.number_input("Tool wear [min]", value=10.0)

if st.button("Predict Failure"):
    try:
        # Encode Type
        type_encoded = oe.transform([[selected_type]]).astype(int)[0][0]
        
        # Prepare features
        input_features = np.array([[type_encoded, air_temp, process_temp, 
                                  rotational_speed, torque, tool_wear]], dtype=np.float32)
        
        # Reshape for deep learning models
        if metadata["model_type"] == "keras" and metadata["input_shape"] == "3D":
            input_features = np.expand_dims(input_features, axis=1)
        
        # Predict
        if metadata["model_type"] == "sklearn":
            pred_class = model.predict(input_features)[0]
        else:
            pred_proba = model.predict(input_features)
            pred_class = np.argmax(pred_proba, axis=1)[0]
        
        # Get result
        result = categories[pred_class] if pred_class < len(categories) else "Unknown"
        st.success(f"Predicted Failure Type: {result}")
    except Exception as e:
        st.error(f"Prediction error: {e}")