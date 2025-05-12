import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Page setup
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
st.title("🔬 Breast Cancer Prediction App")
st.markdown("Enter **31 comma-separated** numerical values representing patient features. The model will predict if the tumor is **Malignant (Cancer)** or **Benign (Not Cancer)**.")

# Input field
example_input = "17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189,0.539"

input_text = st.text_input("📝 Input Features (comma-separated)", value=example_input)

# Prediction logic
if st.button("🔍 Predict"):
    try:
        # Convert and validate input
        input_data = np.fromstring(input_text, sep=',')
        
        if input_data.shape[0] != 31:
            st.error("❌ Please enter exactly 31 numeric values.")
        else:
            # Preprocess using the original scaler
            input_scaled = scaler.transform(input_data.reshape(1, -1))
            
            # Predict
            prediction = model.predict(input_scaled)

            # Output
            if prediction[0] == 1:
                st.success("🛑 Prediction: **Cancer (Malignant)**")
            else:
                st.success("✅ Prediction: **Not Cancer (Benign)**")
    except Exception as e:
        st.error(f"⚠️ Invalid input. Error: {e}")

# Footer
st.markdown("---")
st.markdown("📊 *This model was trained on the Wisconsin Breast Cancer dataset using Logistic Regression.*")
