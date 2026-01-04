import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---- Load pickles ----
model = pickle.load(open("medicine_model.pkl", "rb"))
scaler = pickle.load(open("medicine_scaler.pkl", "rb"))
features = pickle.load(open("medicine_features.pkl", "rb"))

# ---- Load encoders ----
enc_medicine = pickle.load(open("Medicine_Name_label_encoder.pkl", "rb"))
enc_condition = pickle.load(open("Condition_Treated_label_encoder.pkl", "rb"))
enc_gender = pickle.load(open("Gender_label_encoder.pkl", "rb"))
enc_agegroup = pickle.load(open("Age_Group_label_encoder.pkl", "rb"))
enc_source = pickle.load(open("Source_label_encoder.pkl", "rb"))

# ---- Load dataset ----
df = pd.read_csv("medicine_hackathon_ready.csv")

st.set_page_config(page_title="Medicine Truth Label AI", layout="centered")
st.title("🩺 Medicine Truth Label AI")

st.subheader("Enter details manually")

# ---- Direct user inputs ----
user_age = st.number_input("Enter Age", 1, 120, step=1)
user_gender = st.text_input("Enter Gender (Male/Female)")
user_disease = st.text_input("Enter Disease/Condition")
user_medicine = st.text_input("Enter Medicine Name")

# Normalize Gender text safely
if user_gender:
    user_gender = user_gender.strip().capitalize()
    if user_gender not in ["Male", "Female"]:
        st.error("⚠ Please enter gender exactly as: Male or Female")
        st.stop()

# ---- Button action ----
if st.button("Analyze Medicine"):
    try:
        # Encode categorical inputs
        med_encoded = enc_medicine.transform([user_medicine])[0]
        disease_encoded = enc_condition.transform([user_disease])[0]
        gender_encoded = enc_gender.transform([user_gender])[0]

        # Build input for model (stable, not random)
        input_data = []
        for f in features:
            if f == "Medicine_Name":
                input_data.append(med_encoded)
            elif f == "Condition_Treated":
                input_data.append(disease_encoded)
            elif f == "Gender":
                input_data.append(gender_encoded)
            elif f == "Age":
                input_data.append(user_age)
            elif f == "Age_Group":
                input_data.append(enc_agegroup.transform([user_age])[0])
            elif f == "Source":
                input_data.append(enc_source.transform(["Dataset"])[0])
            else:
                input_data.append(df[f].mean())  # only for least impactful numeric fields

        df_input = pd.DataFrame([input_data], columns=features)
        scaled = scaler.transform(df_input)

        # Predict classification (0 or 1)
        pred = model.predict(scaled)[0]

        # ---- Classification result ----
        if pred == 1:
            st.success("Adherence Prediction: Good (Low Risk)")
        else:
            st.error("Adherence Prediction: Poor (High Risk) ⚠")

        # ---- Medicine comparison for same disease ----
        st.subheader(f"💊 Personalized medicine comparison for disease: {user_disease}")

        disease_df = df[df["Condition_Treated"] == user_disease]

        if disease_df.empty:
            st.error("❗ No medicines found for this disease!")
        else:
            comparison = disease_df.groupby("Medicine_Name").agg({
                "Side_Effects_Severity": "mean",
                "Average_Price": "mean",
                "Medicine_Effectiveness_Score": "mean",
                "Dosage_mg": "mean"
            }).round(1)

            comparison.rename(columns={
                "Side_Effects_Severity": "Avg_SideEffect_Risk",
                "Average_Price": "Avg_Price (₹)",
                "Medicine_Effectiveness_Score": "Avg_Effectiveness",
                "Dosage_mg": "Avg_Dosage_mg"
            }, inplace=True)

            # Sort by side effect severity (ascending = safest first)
            comparison = comparison.sort_values(by="Avg_SideEffect_Risk")
            st.dataframe(comparison)

            # Summary for medicines of same disease
            st.subheader("💡 Medicine Summary:")
            for med, row in comparison.iterrows():
                risk = row["Avg_SideEffect_Risk"]
                if risk <= 2:
                    msg = "Mild side effects, safer option"
                elif risk <= 3.5:
                    msg = "Medium side effects, monitor your health"
                else:
                    msg = "Strong side effects, be careful!"
                st.write(f"- {med}: {msg}")

        # ---- Chatbot note ----
        st.subheader("🤖 Lifestyle Recommendations")
        st.info("Lifestyle, diet plans, do’s/don’ts, tips, and motivation will be generated using chatbot later.")

    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")
