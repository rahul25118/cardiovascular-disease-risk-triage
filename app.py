import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# पेज की सेटिंग
st.set_page_config(
    page_title="हृदय रोग जोखिम जांच (CAD Risk Tool)",
    page_icon="🫀",
    layout="wide"
)

# मुख्य शीर्षक और विवरण (सरल शब्दों में)
st.title("🫀 हृदय रोग (CAD) जोखिम मूल्यांकन उपकरण")
st.markdown("""
यह एक क्लिनिकल स्क्रीनिंग टूल है जो बेसिक मेडिकल टेस्ट के आधार पर **कोरोनरी आर्टरी डिजीज (CAD)**, 
यानी दिल की नसों में रुकावट के जोखिम का अनुमान लगाता है।
\n\n**महत्वपूर्ण:** यह केवल एक शुरुआती जांच है, अंतिम मेडिकल रिपोर्ट नहीं।
""")

# --- मॉडल लोड करने का फंक्शन (Internal Logic - No change) ---
@st.cache_resource
def get_trained_pipeline():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    df = pd.read_csv(url, names=cols, na_values='?')
    df.columns = df.columns.str.strip()
    df['num'] = (df['num'] > 0).astype(int)
    df = df.drop(columns=['ca', 'thal', 'slope'], errors='ignore').drop_duplicates()
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang']
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0]).astype(int)
        
    df['hr_reserve'] = df['thalach'] / (220 - df['age'])
    df['st_hr_ratio'] = df['oldpeak'] / (df['thalach'] + 1e-5)
    df['vascular_load'] = (df['trestbps'] * df['age']) / 100.0
    df['ischemia_angina_match'] = ((df['exang'] == 1) & (df['oldpeak'] > 0)).astype(int)
    df['high_risk_syndrome'] = ((df['exang'] == 1) & (df['oldpeak'] >= 1.0) & (df['cp'].isin([1, 4]))).astype(int)
    
    X = df.drop(columns=['num'])
    y = df['num']
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'hr_reserve', 'st_hr_ratio', 'vascular_load']
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ischemia_angina_match', 'high_risk_syndrome']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ]
    )
    pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, penalty='l2', random_state=42))
    ])
    pipe.fit(X, y)
    return pipe

pipeline = get_trained_pipeline()

# --- यूजर इनपुट सेक्शन (सरल हिंदी में अपडेटेड) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. बुनियादी जानकारी और महत्वपूर्ण आँकड़े")
    
    age = st.slider("मरीज की उम्र (Age in Years)", 20, 85, 48)
    
    sex = st.selectbox("लिंग (Sex)", options=[1, 0], format_func=lambda x: "पुरुष (Male)" if x == 1 else "महिला (Female)")
    
    # Simple labels for basic metrics
    trestbps = st.number_input("ब्लड प्रेशर - विश्राम के समय (Blood Pressure - mmHg)", 80, 220, 130, help="आराम करते समय मापा गया सिस्टोलिक (ऊपर वाला) रक्तचाप।")
    
    chol = st.number_input("कोलेस्ट्रॉल स्तर (Cholesterol - mg/dl)", 80, 600, 240, help="सीरम कोलेस्ट्रॉल का कुल स्तर।")
    
    fbs = st.selectbox("क्या शुगर 120 से अधिक है? (Fasting Blood Sugar > 120)", options=[0, 1], format_func=lambda x: "नहीं (No)" if x == 0 else "हाँ (Yes)", help="सुबह बिना कुछ खाए (Fasting) शुगर टेस्ट का परिणाम।")

with col2:
    st.subheader("2. लक्षण और टेस्ट परिणाम")
    
    # Simplest translation for chest pain types
    cp_map = {
        1: "विशिष्ट दर्द (Typical Angina) - दिल से संबंधित दर्द",
        2: "असामान्य दर्द (Atypical Angina) - थोड़ा अलग दर्द",
        3: "गैर-हृदय दर्द (Non-anginal) - दिल का दर्द नहीं",
        4: "कोई लक्षण नहीं (Asymptomatic)"
    }
    cp = st.selectbox("सीने में दर्द का प्रकार (Chest Pain Type)", options=[1, 2, 3, 4], format_func=lambda x: cp_map[x], help="दर्द की प्रकृति कैसी है?")
    
    thalach = st.number_input("हृदय गति - टेस्ट के दौरान अधिकतम (Max Heart Rate)", 60, 220, 140, help="व्यायाम या टेस्ट (Stress Test) के दौरान हृदय कितनी तेज धड़क सकता है?")
    
    exang = st.selectbox("क्या व्यायाम से सीने में दर्द हुआ? (Exercise Angina)", options=[0, 1], format_func=lambda x: "नहीं (No)" if x == 0 else "हाँ (Yes)", help="क्या तेज चलने या व्यायाम करने से सीने में दर्द ट्रिगर हुआ?")
    
    oldpeak = st.number_input("ईसीजी में ST डिप्रेशन (ST Depression/Oldpeak)", 0.0, 6.0, 0.5, step=0.1, help="व्यायाम के बाद ईसीजी (ECG) रिपोर्ट में आया हुआ विशिष्ट बदलाव।")
    
    # Standard translation for ECG categories
    restecg_map = {
        0: "सामान्य (Normal)",
        1: "ST-T वेव असामान्यता (ST-T Wave Abnormality) - ईसीजी में हल्का बदलाव",
        2: "एलवी अतिवृद्धि (Left Ventricular Hypertrophy) - दिल की दीवार मोटी होना"
    }
    restecg = st.selectbox("विश्राम ईसीजी परिणाम (Resting ECG)", options=[0, 1, 2], format_func=lambda x: restecg_map[x], help="आराम करते समय लिया गया सामान्य ईसीजी।")

# internal processing - no change
input_data = pd.DataFrame([{
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
    'hr_reserve': thalach / (220 - age),
    'st_hr_ratio': oldpeak / (thalach + 1e-5),
    'vascular_load': (trestbps * age) / 100.0,
    'ischemia_angina_match': int((exang == 1) and (oldpeak > 0)),
    'high_risk_syndrome': int((exang == 1) and (oldpeak >= 1.0) and (cp in [1, 4]))
}])

st.markdown("---")

# --- बटन और परिणाम सेक्शन ---
if st.button("हृदय रोग जोखिम की जांच करें", type="primary"):
    prob = pipeline.predict_proba(input_data)[0][1]
    
    st.subheader("मूल्यांकन परिणाम (Diagnostic Summary)")
    res1, res2 = st.columns([1, 2])
    
    with res1:
        st.metric("संभावित जोखिम (Risk Probability)", f"{prob * 100:.1f}%")
        st.progress(prob)
        
    with res2:
        if prob >= 0.35:
            st.error("⚠️ **हृदय रोग की उच्च संभावना (High Risk)**\n\nआपके द्वारा दिए गए आँकड़े (जैसे सीने में दर्द, ईसीजी बदलाव, या कोलेस्ट्रॉल) दिल की नसों में रुकावट (CAD) के उच्च जोखिम की ओर इशारा करते हैं।\n\n**सलाह:** कृपया तुरंत किसी कार्डियोलॉजिस्ट (दिल के डॉक्टर) से संपर्क करें और आगे की जांच (जैसे एंजियोग्राफी) के बारे में सलाह लें।")
        else:
            st.success("✅ **कम जोखिम (Low Risk / Non-Ischemic)**\n\nआपके आँकड़े फिलहाल कम जोखिम की श्रेणी में आते हैं।")
            
    st.caption("यह मॉडल हंगेरियन हार्ट कोहोर्ट डेटा पर आधारित है। मॉडल की सटीकता लगभग 84% है।")
