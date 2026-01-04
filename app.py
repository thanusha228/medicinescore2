import streamlit as st
import pickle
import pandas as pd

# Load pickles
model = pickle.load(open("medicine_model.pkl", "rb"))       # classifier
scaler = pickle.load(open("medicine_scaler.pkl", "rb"))
features = pickle.load(open("medicine_features.pkl", "rb"))

# Load individual encoders
enc_medicine = pickle.load(open("enc_medicine.pkl", "rb"))
enc_condition = pickle.load(open("enc_condition.pkl", "rb"))
enc_gender = pickle.load(open("enc_gender.pkl", "rb"))

# Load dataset
df = pd.read_csv("medicine_truth_label_hackathon_updated.csv")

st.set_page_config(page_title="Medicine Truth Label AI", layout="centered")
st.title("🩺 Medicine Truth Label AI")

st.subheader("Enter Patient & Medicine Details")

# Direct user input (no dropdowns)
user_age = st.number_input("Enter Age", min_value=1, max_value=120, step=1)
user_gender = st.text_input("Enter Gender (Male/Female)")
user_disease = st.text_input("Enter Disease/Condition")
user_medicine = st.text_input("Enter Medicine Name")

# Normalize gender input safely
if user_gender:
    user_gender = user_gender.strip().capitalize()
    if user_gender not in ["Male", "Female"]:
        st.error("⚠ Please enter gender exactly as: Male or Female")

# Predict only when disease and medicine are provided
if st.button("Get Medicine Summary"):
    try:
        # Encode categorical user inputs
        med_encoded = enc_medicine.transform([user_medicine])[0]
        disease_encoded = enc_condition.transform([user_disease])[0]
        gender_encoded = enc_gender.transform([user_gender])[0]

        # Build model input only from user-entered values (stable)
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
            else:
                input_data.append(0.0)  # stable default numeric for unused fields

        df_input = pd.DataFrame([input_data], columns=features)
        scaled_input = scaler.transform(df_input)

        # Classification prediction
        pred = model.predict(scaled_input)[0]

        # ---- Medicine Comparison Output (CSV-based) ----
        st.subheader(f"🧾 Medicine Comparison for disease: {user_disease}")

        disease_df = df[df["Condition_Treated"] == user_disease]

        if disease_df.empty:
            st.error("❗ No medicines found for this disease in dataset!")
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

            comparison = comparison.sort_values(by="Avg_SideEffect_Risk")  # safer first
            st.dataframe(comparison)

            # Summary text (changes based on disease medicines)
            st.subheader("💡 Summary for patient:")
            for med, row in comparison.iterrows():
                risk = row["Avg_SideEffect_Risk"]
                if risk <= 2:
                    msg = "mild side effects and safer"
                elif risk <= 3.5:
                    msg = "medium side effects, monitor health"
                else:
                    msg = "strong side effects, be careful!"
                st.write(f"- {med}: {msg}")

        # Adherence classification result
        if pred == 1:
            st.success("Adherence Prediction: Good (Low Risk)")
        else:
            st.error("Adherence Prediction: Poor (High Risk) ⚠")

    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")
