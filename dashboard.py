import io
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Healthcare Demo Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_patients_from_api(base_url: str) -> pd.DataFrame:
    try:
        resp = requests.get(f"{base_url}/patients", timeout=10)
        resp.raise_for_status()
        patients = resp.json()
        df = pd.DataFrame(patients)
        # Ensure datetime parsing for display
        if "last_visit_date" in df.columns:
            df["last_visit_date"] = pd.to_datetime(df["last_visit_date"]).dt.date
        return df
    except Exception as e:
        st.warning(f"Failed to fetch from API: {e}. Falling back to local synthetic data.")
        # Fallback: import generation from server logic by simple HTTP to individual endpoints
        rng = np.random.RandomState(42)
        ids = np.arange(1, 51)
        rows = []
        for pid in ids:
            try:
                r = requests.get(f"{base_url}/patient/{pid}", timeout=2)
                if r.status_code == 200:
                    rows.append(r.json()["patient"])
            except Exception:
                continue
        if rows:
            df = pd.DataFrame(rows)
            if "last_visit_date" in df.columns:
                df["last_visit_date"] = pd.to_datetime(df["last_visit_date"]).dt.date
            return df
        # As a last resort, return empty
        return pd.DataFrame()


api_base = st.secrets.get("API_BASE_URL", "http://127.0.0.1:8000")
patients_df = load_patients_from_api(api_base)


def fetch_briefing(patient_id: int) -> str:
    try:
        resp = requests.get(f"{api_base}/patient/{patient_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("briefing", "")
        return "Backend unavailable or returned an error."
    except Exception:
        return "Backend not reachable. Ensure the API is running or deployed."


st.title("Healthcare Demo Dashboard")

with st.sidebar:
    st.header("Filters")
    cities = ["All"] + sorted(patients_df["city"].unique().tolist())
    selected_city = st.selectbox("City", cities)
    risk_levels = ["All", "High", "Medium", "Low"]
    selected_risk = st.selectbox("Risk level", risk_levels)

filtered_df = patients_df.copy()
if selected_city != "All":
    filtered_df = filtered_df.loc[filtered_df["city"] == selected_city]
if selected_risk != "All":
    filtered_df = filtered_df.loc[filtered_df["risk_level"] == selected_risk]

st.subheader("Patients")
st.dataframe(
    filtered_df[
        [
            "patient_id",
            "name",
            "age",
            "gender",
            "city",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "hba1c_pct",
            "total_cholesterol_mgdl",
            "risk_level",
        ]
    ].sort_values(["risk_level", "name"]),
    use_container_width=True,
)

st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Select Patient")
    patient_options = (
        filtered_df[["patient_id", "name"]]
        .sort_values("name")
        .apply(lambda r: f"{int(r['patient_id'])} - {r['name']}", axis=1)
        .tolist()
    )
    if len(patient_options) == 0:
        st.info("No patients match the selected filters.")
        st.stop()
    selected_option = st.selectbox("Patient", patient_options)
    selected_id = int(selected_option.split(" - ")[0])

    patient_row = patients_df.loc[patients_df["patient_id"] == selected_id].iloc[0]
    st.write(
        f"Age: {int(patient_row['age'])} | Gender: {patient_row['gender']} | City: {patient_row['city']}"
    )
    st.write(
        f"BP: {int(patient_row['systolic_bp'])}/{int(patient_row['diastolic_bp'])} mmHg | HR: {int(patient_row['heart_rate'])} bpm | Temp: {float(patient_row['temperature_c'])} °C"
    )
    st.write(
        f"HbA1c: {float(patient_row['hba1c_pct'])}% | Cholesterol: {int(patient_row['total_cholesterol_mgdl'])} mg/dL"
    )
    st.write(
        f"BMI: {float(patient_row['bmi'])} | Last visit: {patient_row['last_visit_date']} ({int(patient_row['months_since_last_visit'])} months ago)"
    )

with col2:
    st.subheader("AI Clinical Briefing")
    briefing_text = fetch_briefing(selected_id)
    st.write(briefing_text)

st.markdown("---")

st.subheader("BMI History")
months = np.arange(11, -1, -1)
base_bmi = float(patient_row["bmi"]) if "patient_row" in locals() else 25.0
noise = np.random.RandomState(0).normal(0, 0.4, size=12)
bmi_series = (base_bmi + np.linspace(-1.0, 0.5, 12) + noise).round(1)
history_df = pd.DataFrame(
    {"Month": [f"-{m}m" for m in months], "BMI": bmi_series}
)
st.line_chart(history_df.set_index("Month"))

st.markdown("---")

st.subheader("Export High-Risk Patients")
high_risk_df = patients_df.loc[patients_df["high_risk"] == True]

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="high_risk")
    return buffer.getvalue()

excel_data = to_excel_bytes(
    high_risk_df[
        [
            "patient_id",
            "name",
            "age",
            "gender",
            "city",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "hba1c_pct",
            "total_cholesterol_mgdl",
            "months_since_last_visit",
            "risk_level",
        ]
    ]
)

st.download_button(
    label="Download High-Risk Patients (Excel)",
    data=excel_data,
    file_name="high_risk_patients.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


