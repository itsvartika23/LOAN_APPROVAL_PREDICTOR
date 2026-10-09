
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("loan_prediction_model.pkl")

st.title("Loan Approval Prediction")

person_age = st.number_input("Age", 18, 100, 25)
person_gender = st.selectbox("Gender", ["male", "female"])
person_education = st.selectbox(
    "Education",
    ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
)
person_income = st.number_input("Annual Income", min_value=0, value=50000)
person_emp_exp = st.number_input("Employment Experience", min_value=0, value=2)
person_home_ownership = st.selectbox(
    "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
)
loan_amnt = st.number_input("Loan Amount", min_value=1, value=10000)
loan_intent = st.selectbox(
    "Loan Purpose",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
     "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
)
loan_int_rate = st.number_input("Interest Rate", min_value=0.1, value=12.0)
loan_percent_income = (
    loan_amnt / person_income if person_income > 0 else 0
)
cb_person_cred_hist_length = st.number_input(
    "Credit History Length", min_value=0.0, value=5.0
)
credit_score = st.number_input("Credit Score", 300, 850, 650)
previous_loan_defaults_on_file = st.selectbox(
    "Previous Loan Defaults", ["Yes", "No"]
)

if st.button("Predict"):
    if person_income <= 0:
        st.error("Annual income must be greater than zero.")
    else:
        input_data = pd.DataFrame([{
            "person_age": person_age,
            "person_gender": person_gender,
            "person_education": person_education,
            "person_income": person_income,
            "person_emp_exp": person_emp_exp,
            "person_home_ownership": person_home_ownership,
            "loan_amnt": loan_amnt,
            "loan_intent": loan_intent,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "credit_score": credit_score,
            "previous_loan_defaults_on_file": previous_loan_defaults_on_file
        }])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Not Approved")
