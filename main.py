import os
from datetime import date, datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    String,
    Boolean,
    create_engine,
    select,
)
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.pool import NullPool


# -----------------------------
# Synthetic dataset generation
# -----------------------------

RANDOM_SEED = 42
rng = np.random.RandomState(RANDOM_SEED)


def _generate_names(num: int) -> List[str]:
    first_names = [
        "Alex",
        "Jordan",
        "Taylor",
        "Casey",
        "Riley",
        "Avery",
        "Jamie",
        "Morgan",
        "Sam",
        "Cameron",
        "Drew",
        "Quinn",
        "Reese",
        "Hayden",
        "Peyton",
        "Rowan",
        "Emerson",
        "Sawyer",
        "Skyler",
        "Finley",
    ]
    last_names = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Miller",
        "Davis",
        "Garcia",
        "Rodriguez",
        "Wilson",
        "Martinez",
        "Anderson",
        "Taylor",
        "Thomas",
        "Hernandez",
        "Moore",
        "Martin",
        "Jackson",
        "Thompson",
        "White",
    ]
    names: List[str] = []
    for _ in range(num):
        names.append(
            f"{rng.choice(first_names)} {rng.choice(last_names)}"
        )
    return names


def _random_last_visit(max_months_back: int = 24) -> date:
    months_back = int(rng.randint(0, max_months_back + 1))
    # Approximate months as 30 days to avoid external dependencies
    days_back = int(months_back * 30 + rng.randint(0, 30))
    return (date.today() - timedelta(days=days_back))


def _months_since(d: date, ref: Optional[date] = None) -> int:
    if ref is None:
        ref = date.today()
    # Month difference ignoring day-of-month granularity
    return (ref.year - d.year) * 12 + (ref.month - d.month)


def _derive_risks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi"] = (df["weight_kg"] / ((df["height_cm"] / 100) ** 2)).round(1)
    df["months_since_last_visit"] = df["last_visit_date"].apply(_months_since)

    df["is_hypertensive"] = (df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)
    df["is_diabetes"] = df["hba1c_pct"] >= 6.5
    df["is_hyperlipidemia"] = df["total_cholesterol_mgdl"] >= 200

    risk_counts = (
        df[["is_hypertensive", "is_diabetes", "is_hyperlipidemia"]]
        .sum(axis=1)
        .astype(int)
    )
    df["high_risk"] = (risk_counts >= 2) | (df["months_since_last_visit"] > 12)

    def _risk_level(row: pd.Series) -> str:
        count = int(
            row["is_hypertensive"] + row["is_diabetes"] + row["is_hyperlipidemia"]
        )
        if row["months_since_last_visit"] > 12 or count >= 2:
            return "High"
        if count == 1:
            return "Medium"
        return "Low"

    df["risk_level"] = df.apply(_risk_level, axis=1)
    return df


def generate_patients(num_patients: int = 50) -> pd.DataFrame:
    patient_ids = np.arange(1, num_patients + 1)
    names = _generate_names(num_patients)
    genders = rng.choice(["Male", "Female"], size=num_patients, p=[0.48, 0.52])
    ages = rng.randint(20, 86, size=num_patients)

    # Height and weight with reasonable ranges
    heights_cm = rng.normal(loc=170, scale=10, size=num_patients).clip(150, 195).round()
    weights_kg = rng.normal(loc=75, scale=15, size=num_patients).clip(45, 140).round(1)

    cities = [
        "New York",
        "San Francisco",
        "Chicago",
        "Boston",
        "Seattle",
        "Austin",
        "Miami",
        "Denver",
    ]
    city_values = rng.choice(cities, size=num_patients)

    systolic = rng.normal(loc=128, scale=18, size=num_patients).clip(95, 190).round().astype(int)
    diastolic = rng.normal(loc=82, scale=12, size=num_patients).clip(55, 120).round().astype(int)
    heart_rate = rng.normal(loc=74, scale=10, size=num_patients).clip(48, 120).round().astype(int)
    temperature_c = rng.normal(loc=36.8, scale=0.5, size=num_patients).clip(35.5, 39.5).round(1)
    hba1c_pct = rng.normal(loc=5.9, scale=0.9, size=num_patients).clip(4.8, 10.5).round(1)
    total_cholesterol = rng.normal(loc=195, scale=35, size=num_patients).clip(120, 320).round().astype(int)

    last_visits = [
        _random_last_visit(max_months_back=24) for _ in range(num_patients)
    ]

    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "name": names,
            "age": ages,
            "gender": genders,
            "height_cm": heights_cm.astype(int),
            "weight_kg": weights_kg,
            "city": city_values,
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "heart_rate": heart_rate,
            "temperature_c": temperature_c,
            "hba1c_pct": hba1c_pct,
            "total_cholesterol_mgdl": total_cholesterol,
            "last_visit_date": last_visits,
        }
    )
    df = _derive_risks(df)
    return df


# Create dataset at import time so both API and Streamlit can reuse it
patients_df: pd.DataFrame = generate_patients(50)


# -----------------------------
# Database (Neon) setup
# -----------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "")
Base = declarative_base()


class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    height_cm = Column(Integer, nullable=False)
    weight_kg = Column(Float, nullable=False)
    city = Column(String, nullable=False)
    systolic_bp = Column(Integer, nullable=False)
    diastolic_bp = Column(Integer, nullable=False)
    heart_rate = Column(Integer, nullable=False)
    temperature_c = Column(Float, nullable=False)
    hba1c_pct = Column(Float, nullable=False)
    total_cholesterol_mgdl = Column(Integer, nullable=False)
    last_visit_date = Column(Date, nullable=False)

    # Derived flags and metrics persisted for simplicity
    bmi = Column(Float, nullable=False)
    months_since_last_visit = Column(Integer, nullable=False)
    is_hypertensive = Column(Boolean, nullable=False)
    is_diabetes = Column(Boolean, nullable=False)
    is_hyperlipidemia = Column(Boolean, nullable=False)
    high_risk = Column(Boolean, nullable=False)
    risk_level = Column(String, nullable=False)


engine = None
if DATABASE_URL:
    # Serverless-friendly: use NullPool so each invocation opens/closes its own connection.
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,
        pool_pre_ping=True,
        future=True,
    )


def init_db_if_configured() -> None:
    if engine is None:
        return
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        count = session.execute(select(Patient).limit(1)).first()
        if count is None:
            # Seed from generated dataframe
            records = []
            for _, row in patients_df.iterrows():
                records.append(
                    Patient(
                        patient_id=int(row["patient_id"]),
                        name=str(row["name"]),
                        age=int(row["age"]),
                        gender=str(row["gender"]),
                        height_cm=int(row["height_cm"]),
                        weight_kg=float(row["weight_kg"]),
                        city=str(row["city"]),
                        systolic_bp=int(row["systolic_bp"]),
                        diastolic_bp=int(row["diastolic_bp"]),
                        heart_rate=int(row["heart_rate"]),
                        temperature_c=float(row["temperature_c"]),
                        hba1c_pct=float(row["hba1c_pct"]),
                        total_cholesterol_mgdl=int(row["total_cholesterol_mgdl"]),
                        last_visit_date=row["last_visit_date"],
                        bmi=float(row["bmi"]),
                        months_since_last_visit=int(row["months_since_last_visit"]),
                        is_hypertensive=bool(row["is_hypertensive"]),
                        is_diabetes=bool(row["is_diabetes"]),
                        is_hyperlipidemia=bool(row["is_hyperlipidemia"]),
                        high_risk=bool(row["high_risk"]),
                        risk_level=str(row["risk_level"]),
                    )
                )
            session.add_all(records)
            session.commit()


# -----------------------------
# FastAPI models and app
# -----------------------------


class PatientResponse(BaseModel):
    patient_id: int
    name: str
    age: int
    gender: str
    city: str
    height_cm: int
    weight_kg: float
    bmi: float
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    temperature_c: float
    hba1c_pct: float
    total_cholesterol_mgdl: int
    last_visit_date: str
    months_since_last_visit: int
    is_hypertensive: bool
    is_diabetes: bool
    is_hyperlipidemia: bool
    high_risk: bool
    risk_level: str


class PatientWithBriefing(BaseModel):
    patient: PatientResponse
    briefing: str


app = FastAPI(title="Healthcare Demo API", version="1.0.0")


# Initialize DB on startup if configured
init_db_if_configured()

# CORS for dashboard hosted elsewhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_clinical_briefing(row: pd.Series) -> str:
    statements: List[str] = []
    risk_bits: List[str] = []
    if bool(row["is_hypertensive"]):
        risk_bits.append("hypertension")
    if bool(row["is_diabetes"]):
        risk_bits.append("diabetes risk")
    if bool(row["is_hyperlipidemia"]):
        risk_bits.append("hyperlipidemia")

    if len(risk_bits) == 0:
        statements.append(
            f"{row['name']} shows no major risk flags today. Continue routine monitoring and healthy lifestyle guidance."
        )
    else:
        flags = ", ".join(risk_bits)
        statements.append(
            f"Key risk considerations: {flags}. Review vitals and labs to optimize management."
        )

    if int(row["months_since_last_visit"]) > 12:
        statements.append(
            "Patient is overdue for follow-up; schedule an appointment and repeat labs."
        )
    else:
        statements.append(
            "Maintain regular follow-up and reinforce diet, exercise, and medication adherence as appropriate."
        )

    return " " .join(statements)


def _row_to_patient_response(row: pd.Series) -> PatientResponse:
    return PatientResponse(
        patient_id=int(row["patient_id"]),
        name=str(row["name"]),
        age=int(row["age"]),
        gender=str(row["gender"]),
        city=str(row["city"]),
        height_cm=int(row["height_cm"]),
        weight_kg=float(row["weight_kg"]),
        bmi=float(row["bmi"]),
        systolic_bp=int(row["systolic_bp"]),
        diastolic_bp=int(row["diastolic_bp"]),
        heart_rate=int(row["heart_rate"]),
        temperature_c=float(row["temperature_c"]),
        hba1c_pct=float(row["hba1c_pct"]),
        total_cholesterol_mgdl=int(row["total_cholesterol_mgdl"]),
        last_visit_date=str(row["last_visit_date"]),
        months_since_last_visit=int(row["months_since_last_visit"]),
        is_hypertensive=bool(row["is_hypertensive"]),
        is_diabetes=bool(row["is_diabetes"]),
        is_hyperlipidemia=bool(row["is_hyperlipidemia"]),
        high_risk=bool(row["high_risk"]),
        risk_level=str(row["risk_level"]),
    )


@app.get("/patient/{patient_id}", response_model=PatientWithBriefing)
def get_patient(patient_id: int) -> PatientWithBriefing:
    if engine is not None:
        with Session(engine) as session:
            patient = session.get(Patient, patient_id)
            if patient is None:
                raise HTTPException(status_code=404, detail="Patient not found")
            # Convert ORM object to pandas Series-like for reuse
            row = pd.Series({
                "patient_id": patient.patient_id,
                "name": patient.name,
                "age": patient.age,
                "gender": patient.gender,
                "city": patient.city,
                "height_cm": patient.height_cm,
                "weight_kg": patient.weight_kg,
                "bmi": patient.bmi,
                "systolic_bp": patient.systolic_bp,
                "diastolic_bp": patient.diastolic_bp,
                "heart_rate": patient.heart_rate,
                "temperature_c": patient.temperature_c,
                "hba1c_pct": patient.hba1c_pct,
                "total_cholesterol_mgdl": patient.total_cholesterol_mgdl,
                "last_visit_date": patient.last_visit_date,
                "months_since_last_visit": patient.months_since_last_visit,
                "is_hypertensive": patient.is_hypertensive,
                "is_diabetes": patient.is_diabetes,
                "is_hyperlipidemia": patient.is_hyperlipidemia,
                "high_risk": patient.high_risk,
                "risk_level": patient.risk_level,
            })
    else:
        match = patients_df.loc[patients_df["patient_id"] == patient_id]
        if match.empty:
            raise HTTPException(status_code=404, detail="Patient not found")
        row = match.iloc[0]
    patient_model = _row_to_patient_response(row)
    briefing = generate_clinical_briefing(row)
    return PatientWithBriefing(patient=patient_model, briefing=briefing)


# Optional convenience root endpoint
@app.get("/")
def root() -> dict:
    return {
        "message": "Healthcare Demo API. Use /patient/{id} to retrieve patient KPIs and briefing.",
        "total_patients": int(patients_df.shape[0]) if engine is None else 50,
    }


@app.get("/patients")
def list_patients() -> List[PatientResponse]:
    rows: List[pd.Series] = []
    if engine is not None:
        with Session(engine) as session:
            for p in session.execute(select(Patient)).scalars().all():
                rows.append(pd.Series({
                    "patient_id": p.patient_id,
                    "name": p.name,
                    "age": p.age,
                    "gender": p.gender,
                    "city": p.city,
                    "height_cm": p.height_cm,
                    "weight_kg": p.weight_kg,
                    "bmi": p.bmi,
                    "systolic_bp": p.systolic_bp,
                    "diastolic_bp": p.diastolic_bp,
                    "heart_rate": p.heart_rate,
                    "temperature_c": p.temperature_c,
                    "hba1c_pct": p.hba1c_pct,
                    "total_cholesterol_mgdl": p.total_cholesterol_mgdl,
                    "last_visit_date": p.last_visit_date,
                    "months_since_last_visit": p.months_since_last_visit,
                    "is_hypertensive": p.is_hypertensive,
                    "is_diabetes": p.is_diabetes,
                    "is_hyperlipidemia": p.is_hyperlipidemia,
                    "high_risk": p.high_risk,
                    "risk_level": p.risk_level,
                }))
    else:
        for _, r in patients_df.iterrows():
            rows.append(r)
    return [_row_to_patient_response(r) for r in rows]


