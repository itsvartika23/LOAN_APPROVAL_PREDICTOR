from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Loan Approval Predictor")
model = joblib.load("loan_prediction_model.pkl")

class LoanInput(BaseModel):
    person_age: float = Field(ge=18, le=100)
    person_gender: Literal["male", "female"]
    person_education: Literal["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    person_income: float = Field(gt=0)
    person_emp_exp: float = Field(ge=0)
    person_home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    loan_amnt: float = Field(gt=0)
    loan_intent: Literal["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    loan_int_rate: float = Field(gt=0, le=40)
    cb_person_cred_hist_length: float = Field(ge=0)
    credit_score: float = Field(ge=300, le=850)
    previous_loan_defaults_on_file: Literal["Yes", "No"]

@app.get("/")
def home():
    return FileResponse("templates/index.html")

@app.post("/predict")
def predict(data: LoanInput):
    row = data.model_dump()

    if row["person_emp_exp"] > row["person_age"]:
        raise HTTPException(status_code=400, detail="Experience cannot be more than age.")

    row["loan_percent_income"] = round(row["loan_amnt"] / row["person_income"], 2)

    df = pd.DataFrame([row])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {
        "approved": prediction == 1,
        "probability": round(probability * 100, 1),
    }
