import os
import random
from datetime import date, timedelta
from typing import List, Optional, Dict, Any
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
rng = random.Random(RANDOM_SEED)


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
        names.append(f"{rng.choice(first_names)} {rng.choice(last_names)}")
    return names


def _random_last_visit(max_months_back: int = 24) -> date:
    months_back = int(rng.randint(0, max_months_back))
    # Approximate months as 30 days
    days_back = int(months_back * 30 + rng.randint(0, 29))
    return date.today() - timedelta(days=days_back)


def _months_since(d: date, ref: Optional[date] = None) -> int:
    if ref is None:
        ref = date.today()
    # Month difference ignoring day-of-month granularity
    return (ref.year - d.year) * 12 + (ref.month - d.month)


def _compute_and_flags(row: Dict[str, Any]) -> Dict[str, Any]:
    height_m = row["height_cm"] / 100.0
    bmi = round(row["weight_kg"] / (height_m * height_m), 1)
    months_since = _months_since(row["last_visit_date"])
    is_hypertensive = row["systolic_bp"] >= 140 or row["diastolic_bp"] >= 90
    is_diabetes = row["hba1c_pct"] >= 6.5
    is_hyperlipidemia = row["total_cholesterol_mgdl"] >= 200
    risk_count = int(is_hypertensive) + int(is_diabetes) + int(is_hyperlipidemia)
    high_risk = risk_count >= 2 or months_since > 12
    if high_risk:
        risk_level = "High"
    elif risk_count == 1:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    row.update(
        {
            "bmi": float(bmi),
            "months_since_last_visit": int(months_since),
            "is_hypertensive": bool(is_hypertensive),
            "is_diabetes": bool(is_diabetes),
            "is_hyperlipidemia": bool(is_hyperlipidemia),
            "high_risk": bool(high_risk),
            "risk_level": risk_level,
        }
    )
    return row


def generate_patients(num_patients: int = 50) -> List[Dict[str, Any]]:
    names = _generate_names(num_patients)
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
    patients: List[Dict[str, Any]] = []
    for i in range(num_patients):
        height_cm = int(max(150, min(195, rng.gauss(170, 10))))
        weight_kg = round(max(45.0, min(140.0, rng.gauss(75, 15))), 1)
        row: Dict[str, Any] = {
            "patient_id": i + 1,
            "name": names[i],
            "age": rng.randint(20, 85),
            "gender": rng.choice(["Male", "Female"]),
            "height_cm": height_cm,
            "weight_kg": float(weight_kg),
            "city": rng.choice(cities),
            "systolic_bp": int(max(95, min(190, round(rng.gauss(128, 18))))),
            "diastolic_bp": int(max(55, min(120, round(rng.gauss(82, 12))))),
            "heart_rate": int(max(48, min(120, round(rng.gauss(74, 10))))),
            "temperature_c": round(max(35.5, min(39.5, rng.gauss(36.8, 0.5))), 1),
            "hba1c_pct": round(max(4.8, min(10.5, rng.gauss(5.9, 0.9))), 1),
            "total_cholesterol_mgdl": int(max(120, min(320, round(rng.gauss(195, 35))))),
            "last_visit_date": _random_last_visit(24),
        }
        patients.append(_compute_and_flags(row))
    return patients


# Create dataset at import time so both API and Streamlit can reuse it
patients_data: List[Dict[str, Any]] = generate_patients(50)


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
            # Seed from generated list
            records = []
            for row in patients_data:
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


def generate_clinical_briefing(row: Dict[str, Any]) -> str:
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

    return " ".join(statements)


def _row_to_patient_response(row: Dict[str, Any]) -> PatientResponse:
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
            row = {
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
            }
    else:
        row = next((p for p in patients_data if p["patient_id"] == patient_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail="Patient not found")
    patient_model = _row_to_patient_response(row)
    briefing = generate_clinical_briefing(row)
    return PatientWithBriefing(patient=patient_model, briefing=briefing)


# Optional convenience root endpoint
@app.get("/")
def root() -> dict:
    return {
        "message": "Healthcare Demo API. Use /patient/{id} to retrieve patient KPIs and briefing.",
        "total_patients": len(patients_data) if engine is None else 50,
    }


@app.get("/patients")
def list_patients() -> List[PatientResponse]:
    rows: List[Dict[str, Any]] = []
    if engine is not None:
        with Session(engine) as session:
            for p in session.execute(select(Patient)).scalars().all():
                rows.append({
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
                })
    else:
        rows = list(patients_data)
    return [_row_to_patient_response(r) for r in rows]


