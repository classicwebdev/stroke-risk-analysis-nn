import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib



@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('stroke_nn_model.h5')
    scaler = joblib.load('stroke_scaler.pkl')
    return model, scaler


model, scaler = load_assets()

st.title("🧠 Stroke Risk Analysis (Neural Network Edition)")
st.write("Enter patient data to see the deep learning risk assessment.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    hypertension = st.selectbox("Hypertension (1=Yes, 0=No)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (1=Yes, 0=No)", [0, 1])
    married = st.selectbox("Ever Married?", ["Yes", "No"])

with col2:
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose = st.number_input("Average Glucose Level", value=100.0)
    bmi = st.number_input("BMI", value=25.0)
    work = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])


if st.button("Analyze Risk"):
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    residence_val = 1 if residence == "Urban" else 0


    work_list = ["Govt_job", "Never_worked", "Private", "Self-employed", "children"]
    work_types = {w: (1.0 if work == w else 0.0) for w in work_list}


    smoke_list = ["Unknown", "formerly smoked", "never smoked", "smokes"]
    smokings = {s: (1.0 if smoking == s else 0.0) for s in smoke_list}


    features = [
        float(gender_val), float(age), float(hypertension), float(heart_disease),
        float(married_val), float(residence_val), float(avg_glucose), float(bmi),
        work_types["Govt_job"], work_types["Never_worked"],
        work_types["Private"], work_types["Self-employed"],
        work_types["children"],
        smokings["Unknown"], smokings["formerly smoked"],
        smokings["never smoked"], smokings["smokes"]
    ]

    features_scaled = scaler.transform([features])
    prediction_proba = model.predict(features_scaled)[0][0]
    is_high_risk = prediction_proba > 0.5

    st.divider()
    st.subheader("Results & Clinical Reasoning")
    if is_high_risk:
        st.error(f"⚠️ HIGH RISK: Neural Network predicts a {prediction_proba:.2%} probability of stroke.")


        reasons = []
        if age > 60: reasons.append("Advanced Age")
        if hypertension == 1: reasons.append("Hypertension")
        if heart_disease == 1: reasons.append("Heart Disease")
        if avg_glucose > 150: reasons.append("High Glucose Levels")
        if smoking == "smokes": reasons.append("Active Smoking Status")

        reason_text = ", ".join(reasons) if reasons else "a combination of multiple clinical factors"
        st.write(f"**Why this prediction?** The AI identified significant influence from: **{reason_text}**.")

    else:
        st.success(f"✅ LOW RISK: Neural Network predicts a {prediction_proba:.2%} probability of stroke.")
        st.write(
            "**Why this prediction?** The AI found that your clinical markers are currently balanced, keeping the risk minimal.")


    st.subheader("📊 Patient Feature Intensity")
    st.write("This chart shows the scaled values of the patient's data.")

    feature_names = [
        "Gender", "Age", "Hypertension", "Heart Disease",
        "Married", "Residence", "Glucose", "BMI",
        "Work_Govt", "Work_Never", "Work_Private", "Work_Self", "Work_Children",
        "Smoke_Unknown", "Smoke_Former", "Smoke_Never", "Smoke_Active"
    ]

    chart_data = pd.DataFrame({
        "Feature": feature_names,
        "Intensity": features_scaled[0]
    }).set_index("Feature")

    st.bar_chart(chart_data)