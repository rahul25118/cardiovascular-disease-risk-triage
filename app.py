import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Cardiovascular Disease Risk Triage",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Cardiovascular Risk Assessment & Screening Tool")
st.markdown("Clinical triage model benchmarked on Hungarian Heart Cohort data (ROC-AUC: 0.897, Sensitivity: 73%).")

@st.cache_resource
def load_pipeline():
    return joblib.load('heart_disease_pipeline.pkl')

pipeline = load_pipeline()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Baseline Vitals")
    age = st.slider("Age (Years)", 20, 85, 48)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 130)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 80, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No (<= 120)" if x == 0 else "Yes (> 120)")

with col2:
    st.subheader("Exercise & Cardiographic Metrics")
    cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4],
                      format_func=lambda x: {1: "Typical Angina (1)", 2: "Atypical Angina (2)", 3: "Non-anginal Pain (3)", 4: "Asymptomatic (4)"}[x])
    thalach = st.number_input("Maximum Achieved Heart Rate (thalach)", 60, 220, 140)
    exang = st.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 0.5, step=0.1)
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                           format_func=lambda x: {0: "Normal (0)", 1: "ST-T Wave Abnormality (1)", 2: "Left Ventricular Hypertrophy (2)"}[x])

# Feature Engineering
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'hr_reserve': thalach / (220 - age),
    'st_hr_ratio': oldpeak / (thalach + 1e-5),
    'vascular_load': (trestbps * age) / 100.0,
    'ischemia_angina_match': int((exang == 1) and (oldpeak > 0)),
    'high_risk_syndrome': int((exang == 1) and (oldpeak >= 1.0) and (cp in [1, 4]))
}])

st.markdown("---")

if st.button("Evaluate Patient Risk", type="primary"):
    prob = pipeline.predict_proba(input_data)[0][1]
    
    st.subheader("Diagnostic Risk Stratification")
    res1, res2 = st.columns([1, 2])
    
    with res1:
        st.metric("Predicted CAD Risk", f"{prob * 100:.1f}%")
        st.progress(prob)
        
    with res2:
        if prob >= 0.35:
            st.error("⚠️ **High Probability of Coronary Artery Disease Detected**\n\nPatient exhibits significant ischemic markers. Recommended for secondary cardiac evaluation / angiography.")
        else:
            st.success("✅ **Low Risk / Non-Ischemic Parameters**\n\nCardiovascular profile aligns within low-risk clinical thresholds.")
            
    st.caption("Model: Regularized L2 Logistic Regression. Out-of-fold generalization accuracy: 82.3% - 84.0%.")
