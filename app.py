import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# model path
MODEL_PATH = Path("models/xgboost_tuned.pkl")

# Load model
@st.cache_resource  # cache so it doesn't reload every time
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Streamlit app
st.title("🎓 EduLens Student Dropout Risk Predictor")

st.write("Enter student details below to predict dropout risk:")

# Collect inputs from user
attendance = st.number_input("Attendance Percentage", min_value=0, max_value=100, value=85)
test_score = st.number_input("Test Score", min_value=0, max_value=100, value=70)
discipline_count = st.number_input("Discipline Count", min_value=0, value=0)
parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])

# Convert parental involvement into numbers (same encoding as training)
pi_map = {"Low": 0, "Medium": 1, "High": 2}
parental_involvement = pi_map[parental_involvement]

# Make dataframe for prediction
input_data = pd.DataFrame([{
    "attendance_percent": attendance,
    "test_score": test_score,
    "discipline_count": discipline_count,
    "parental_involvement": parental_involvement
}])

# Predict
if st.button("Predict Dropout Risk"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Dropout Risk!")
    else:
        st.success("✅ Low Dropout Risk")
