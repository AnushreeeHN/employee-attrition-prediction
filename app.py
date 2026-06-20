import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load('attrition_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title("Employee Attrition Predictor")
st.write("Enter employee details to estimate attrition risk.")

# --- Collect key inputs from the user ---
age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
distance_from_home = st.slider("Distance From Home (km)", 1, 30, 5)
total_working_years = st.slider("Total Working Years", 0, 40, 5)
years_at_company = st.slider("Years At Company", 0, 40, 3)
job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
work_life_balance = st.selectbox("Work Life Balance (1=Bad, 4=Best)", [1, 2, 3, 4])
overtime = st.selectbox("OverTime", ["Yes", "No"])
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
job_role = st.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# --- Build a single-row dataframe with ALL expected columns, default 0 ---
input_dict = {col: 0 for col in feature_columns}

# Fill in the numeric ones directly
input_dict['Age'] = age
input_dict['MonthlyIncome'] = monthly_income
input_dict['DistanceFromHome'] = distance_from_home
input_dict['TotalWorkingYears'] = total_working_years
input_dict['YearsAtCompany'] = years_at_company
input_dict['JobSatisfaction'] = job_satisfaction
input_dict['WorkLifeBalance'] = work_life_balance

# Fill in the one-hot encoded categorical ones (only set to 1 if column exists)
overtime_col = 'OverTime_Yes'
if overtime_col in input_dict and overtime == "Yes":
    input_dict[overtime_col] = 1

dept_col = f'Department_{department}'
if dept_col in input_dict:
    input_dict[dept_col] = 1

role_col = f'JobRole_{job_role}'
if role_col in input_dict:
    input_dict[role_col] = 1

marital_col = f'MaritalStatus_{marital_status}'
if marital_col in input_dict:
    input_dict[marital_col] = 1

# --- Convert to dataframe in the correct column order ---
input_df = pd.DataFrame([input_dict])[feature_columns]

# --- Predict ---
if st.button("Predict Attrition Risk"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ High risk of attrition — Probability: {probability:.1%}")
    else:
        st.success(f"✅ Low risk of attrition — Probability: {probability:.1%}")