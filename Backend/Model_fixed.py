"""
main.py  —  NutriSense AI  |  Deficiency Diagnosis API
--------------------------------------------------------
Stack : FastAPI + asyncpg + bcrypt + python-jose (JWT) + scikit-learn

Install:
    pip install fastapi uvicorn asyncpg bcrypt "python-jose[cryptography]" python-dotenv "pydantic[email]" scikit-learn numpy

Run:
    uvicorn main:app --reload --port 8000

Frontend-compatible endpoints:
    POST  /signup                       – register (matches signup.html)
    POST  /login                        – login, returns {success, user, token}
    POST  /assessments/diagnose         – one-shot diagnose from nc_final payload
    POST  /assessments/{id}/diagnose    – diagnose from saved assessment (result.html fallback)
"""

import os
import pickle
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

import asyncpg
import bcrypt
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

SECRET_KEY   = os.getenv("SECRET_KEY", "nutrisense-secret-key-change-in-production")
ALGORITHM    = "HS256"
TOKEN_EXPIRE = int(os.getenv("TOKEN_EXPIRE_MINUTES", 60))

DB_DSN = (
    f"postgresql://{os.getenv('DB_USER','postgres')}:"
    f"{os.getenv('DB_PASSWORD','')}@"
    f"{os.getenv('DB_HOST','localhost')}:"
    f"{os.getenv('DB_PORT','5432')}/"
    f"{os.getenv('DB_NAME','deficiency_db')}"
)

# ── Load ML model ──────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "nutriscreen_model.pkl")
LE_PATH    = os.path.join(BASE_DIR, "label_encoder.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    print("✅ nutriscreen_model.pkl loaded")
except Exception as e:
    print(f"⚠️  Model load failed: {e}")
    _model = None

try:
    with open(LE_PATH, "rb") as f:
        _le = pickle.load(f)
    print("✅ label_encoder.pkl loaded")
except Exception as e:
    print(f"⚠️  Label encoder load failed: {e}")
    _le = None

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="NutriSense AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login", auto_error=False)

# ── Serve frontend ─────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

@app.get("/")
async def root():
    p = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(p) if os.path.exists(p) else {"status": "NutriSense API running"}

@app.get("/login")
async def serve_login():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"))

@app.get("/signup")
async def serve_signup():
    return FileResponse(os.path.join(FRONTEND_DIR, "signup.html"))

@app.get("/home")
async def serve_home():
    return FileResponse(os.path.join(FRONTEND_DIR, "home.html"))

@app.get("/result")
async def serve_result():
    return FileResponse(os.path.join(FRONTEND_DIR, "result.html"))

@app.get("/user_basic")
async def serve_basic():
    return FileResponse(os.path.join(FRONTEND_DIR, "user_basic.html"))

@app.get("/user_lifestyle")
async def serve_lifestyle():
    return FileResponse(os.path.join(FRONTEND_DIR, "user_lifestyle.html"))

@app.get("/user_symptoms")
async def serve_symptoms():
    return FileResponse(os.path.join(FRONTEND_DIR, "user_symptoms.html"))

# Also serve .html extensions directly
@app.get("/{page}.html")
async def serve_html(page: str):
    p = os.path.join(FRONTEND_DIR, f"{page}.html")
    if os.path.exists(p):
        return FileResponse(p)
    raise HTTPException(404, "Page not found")

# ── DB pool ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    try:
        app.state.pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=10)
        print("✅ Database connected")
    except Exception as e:
        print(f"⚠️  DB unavailable: {e}. Running in predict-only mode.")
        app.state.pool = None

@app.on_event("shutdown")
async def shutdown():
    if getattr(app.state, "pool", None):
        await app.state.pool.close()

async def get_db():
    pool = getattr(app.state, "pool", None)
    if not pool:
        yield None
        return
    async with pool.acquire() as conn:
        yield conn

# ── JWT helpers ────────────────────────────────────────────────────────────────

def create_token(user_id: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE)
    return jwt.encode({"sub": str(user_id), "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None

async def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    if not token:
        raise HTTPException(401, "Not authenticated")
    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(401, "Invalid or expired token")
    if db is None:
        return {"user_id": user_id, "username": "user", "full_name": "User", "role": "patient"}
    row = await db.fetchrow("SELECT * FROM users WHERE user_id=$1 AND is_active=TRUE", user_id)
    if not row:
        raise HTTPException(401, "User not found")
    return dict(row)

# ── Password helpers ───────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

# ═══════════════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class SignupRequest(BaseModel):
    name:     str = Field(..., min_length=2)
    email:    EmailStr
    password: str = Field(..., min_length=6)

class LoginRequest(BaseModel):
    email:    str
    password: str

class DiagnoseRequest(BaseModel):
    # Basic
    Age:                            Optional[float] = None
    BMI:                            Optional[float] = None
    Gender:                         Optional[str]   = None
    Geographic_Region:              Optional[str]   = None
    Diet_Type:                      Optional[str]   = None
    Family_History_of_Deficiency:   Optional[str]   = None
    # Lifestyle
    Pregnancy_Status:               Optional[str]   = None
    Smoking_Status:                 Optional[str]   = None
    Alcohol_Consumption:            Optional[str]   = None
    Sun_Exposure:                   Optional[str]   = None
    Physical_Activity_Level:        Optional[str]   = None
    Stress_Level:                   Optional[str]   = None
    Chronic_Condition:              Optional[str]   = None
    Medication_Use:                 Optional[str]   = None
    Recent_Blood_Loss:              Optional[str]   = None
    # Symptoms
    Digestive_Issues:               Optional[str]   = None
    Energy_Level:                   Optional[str]   = None
    Sleep_Quality:                  Optional[str]   = None
    Fatigue:                        Optional[str]   = None
    Pallor:                         Optional[str]   = None
    Shortness_of_Breath:            Optional[str]   = None
    Cold_Intolerance:               Optional[str]   = None
    Bone_Pain:                      Optional[str]   = None
    Muscle_Weakness:                Optional[str]   = None
    Frequent_Illness:               Optional[str]   = None
    Hair_Loss:                      Optional[str]   = None
    Tingling_Numbness:              Optional[str]   = None
    Memory_Cognitive_Issues:        Optional[str]   = None
    Balance_Problems:               Optional[str]   = None
    Mood_Swings:                    Optional[str]   = None
    Muscle_Cramps:                  Optional[str]   = None
    Dental_Issues:                  Optional[str]   = None
    Brittle_Nails:                  Optional[str]   = None
    Anxiety_and_Poor_Sleep:         Optional[str]   = None
    Frequent_Headaches:             Optional[str]   = None
    Loss_of_Appetite:               Optional[str]   = None
    Slow_Wound_Healing:             Optional[str]   = None
    Loss_of_Taste_Smell:            Optional[str]   = None
    Mouth_Sores:                    Optional[str]   = None
    Tongue_Swelling:                Optional[str]   = None
    Restless_Legs:                  Optional[str]   = None
    Night_Sweats:                   Optional[str]   = None

# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/signup", status_code=201)
async def signup(body: SignupRequest, db=Depends(get_db)):
    """Register a new user. Returns {success, user, token}."""
    import hashlib

    if db is None:
        fake_id = hashlib.md5(body.email.encode()).hexdigest()
        token   = create_token(fake_id)
        return {"success": True, "token": token,
                "user": {"id": fake_id, "name": body.name, "email": body.email}}

    existing = await db.fetchrow("SELECT user_id FROM users WHERE email=$1", body.email)
    if existing:
        raise HTTPException(400, "Email already registered.")

    username = body.email.split("@")[0]
    count    = await db.fetchval(
        "SELECT COUNT(*) FROM users WHERE username LIKE $1", f"{username}%"
    )
    if count:
        username = f"{username}{count}"

    pw_hash = hash_password(body.password)
    user_id = await db.fetchval(
        "INSERT INTO users (username, email, password_hash, full_name) VALUES ($1,$2,$3,$4) RETURNING user_id",
        username, body.email, pw_hash, body.name
    )
    token = create_token(str(user_id))
    return {"success": True, "token": token,
            "user": {"id": str(user_id), "name": body.name, "email": body.email}}


@app.post("/login")
async def login(body: LoginRequest, db=Depends(get_db)):
    """Authenticate user. Returns {success, user: {name, email}, token}."""
    import hashlib

    if db is None:
        fake_id = hashlib.md5(body.email.encode()).hexdigest()
        token   = create_token(fake_id)
        name    = body.email.split("@")[0].title()
        return {"success": True, "token": token,
                "user": {"id": fake_id, "name": name, "email": body.email}}

    row = await db.fetchrow(
        "SELECT * FROM users WHERE email=$1 AND is_active=TRUE", body.email
    )
    if not row or not verify_password(body.password, row["password_hash"]):
        return {"success": False, "error": "Invalid email or password"}

    await db.execute("UPDATE users SET last_login_at=NOW() WHERE user_id=$1", row["user_id"])
    token = create_token(str(row["user_id"]))
    return {
        "success": True,
        "token":   token,
        "user": {
            "id":    str(row["user_id"]),
            "name":  row["full_name"] or row["username"],
            "email": row["email"]
        }
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSE  (one-shot — primary route used by result.html)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/assessments/diagnose")
async def diagnose_oneshot(body: DiagnoseRequest, request: Request, db=Depends(get_db)):
    """
    Receives the merged nc_final payload, runs the ML model, returns prediction.
    Optionally saves to DB when user token is present.
    """
    d = body.model_dump()
    predicted, confidence = _run_model(d)

    assessment_id = None
    if db is not None:
        try:
            auth = request.headers.get("Authorization", "")
            user_id = decode_token(auth[7:]) if auth.startswith("Bearer ") else None
            if user_id:
                assessment_id = await db.fetchval(
                    "INSERT INTO assessments (user_id, status) VALUES ($1,'completed') RETURNING assessment_id",
                    user_id
                )
                await db.execute(
                    """INSERT INTO diagnoses (assessment_id, predicted_deficiency, confidence_score, diagnosis_source)
                       VALUES ($1, $2::deficiency_type, $3, 'model')""",
                    str(assessment_id), predicted, confidence
                )
        except Exception as e:
            print(f"DB save skipped: {e}")

    return {
        "predicted_deficiency": predicted,
        "confidence_score":     round(confidence, 4),
        "assessment_id":        str(assessment_id) if assessment_id else None,
    }


@app.post("/assessments/{assessment_id}/diagnose")
async def diagnose_from_saved(
    assessment_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Fallback route: reads saved risk_factors + symptoms from DB and runs model."""
    if db is None:
        return {"predicted_deficiency": "Iron", "confidence_score": 0.75}

    rf  = await db.fetchrow("SELECT * FROM risk_factors WHERE assessment_id=$1", assessment_id)
    sym = await db.fetchrow("SELECT * FROM symptoms    WHERE assessment_id=$1", assessment_id)

    if not rf or not sym:
        raise HTTPException(400, "Submit risk factors and symptoms first.")

    predicted, confidence = _run_model({**dict(rf), **dict(sym)})

    await db.execute(
        """INSERT INTO diagnoses (assessment_id, predicted_deficiency, confidence_score, diagnosis_source)
           VALUES ($1, $2::deficiency_type, $3, 'model') ON CONFLICT DO NOTHING""",
        assessment_id, predicted, confidence
    )
    return {"predicted_deficiency": predicted, "confidence_score": round(confidence, 4)}

# ═══════════════════════════════════════════════════════════════════════════════
#  ML FEATURE BUILDER & RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd

# Exact feature order the RandomForestClassifier was trained on
_FEATURE_NAMES = None  # loaded after model is confirmed

_SYM_MAP      = {"No": 0, "Occasional": 1, "Yes": 2}
_BOOL_MAP     = {"Yes": 1, "No": 0, "True": 1, "False": 0, True: 1, False: 0}
_SUN_MAP      = {"Minimal": 0, "Moderate": 1, "High": 2}
_ACTIVITY_MAP = {"Sedentary": 0, "Moderate": 1, "Active": 2, "Very Active": 3}
_STRESS_MAP   = {"Low": 0, "Moderate": 1, "High": 2}
_ENERGY_MAP   = {"Low": 0, "Medium": 1, "High": 2}
_SLEEP_MAP    = {"Poor": 0, "Fair": 1, "Average": 1, "Good": 2, "Excellent": 3}

if _model is not None:
    _FEATURE_NAMES = list(_model.feature_names_in_)


def _g(d, key):
    """Lookup by key or its lowercase variant."""
    return d.get(key) or d.get(key.lower()) or ""


def _build_features(d: dict) -> "pd.DataFrame":
    """
    Build the exact 68-feature DataFrame the RandomForest expects.
    Handles both Title_Case keys (from nc_final) and snake_case (from DB).
    """
    row = {}

    # ── Ordinal / numeric (features 0-33) ──────────────────────────────────
    row["Age"]                          = float(_g(d, "Age")  or 30)
    row["BMI"]                          = float(_g(d, "BMI")  or 22)
    row["Sun_Exposure"]                 = _SUN_MAP.get(_g(d, "Sun_Exposure"), 1)
    row["Physical_Activity_Level"]      = _ACTIVITY_MAP.get(_g(d, "Physical_Activity_Level"), 1)
    row["Stress_Level"]                 = _STRESS_MAP.get(_g(d, "Stress_Level"), 1)
    row["Recent_Blood_Loss"]            = _BOOL_MAP.get(_g(d, "Recent_Blood_Loss"), 0)
    row["Family_History_of_Deficiency"] = _BOOL_MAP.get(_g(d, "Family_History_of_Deficiency"), 0)
    row["Digestive_Issues"]             = _BOOL_MAP.get(_g(d, "Digestive_Issues"), 0)

    for sym in ["Fatigue","Pallor","Shortness_of_Breath","Cold_Intolerance","Bone_Pain",
                "Muscle_Weakness","Frequent_Illness","Hair_Loss","Tingling_Numbness",
                "Memory_Cognitive_Issues","Balance_Problems","Mood_Swings","Muscle_Cramps",
                "Dental_Issues","Brittle_Nails","Anxiety_and_Poor_Sleep","Frequent_Headaches",
                "Loss_of_Appetite","Slow_Wound_Healing","Loss_of_Taste_Smell","Mouth_Sores",
                "Tongue_Swelling","Restless_Legs","Night_Sweats"]:
        row[sym] = _SYM_MAP.get(_g(d, sym), 0)

    row["Energy_Level"] = _ENERGY_MAP.get(_g(d, "Energy_Level"), 1)
    row["Sleep_Quality"] = _SLEEP_MAP.get(_g(d, "Sleep_Quality"), 1)

    # ── One-hot: Gender ─────────────────────────────────────────────────────
    gender = _g(d, "Gender")
    row["Gender_Female"] = 1 if gender == "Female" else 0
    row["Gender_Male"]   = 1 if gender == "Male"   else 0
    row["Gender_Other"]  = 1 if gender == "Other"  else 0

    # ── One-hot: Geographic_Region ──────────────────────────────────────────
    # Note: model uses "Subtropical" not "Arid/Desert" — map frontend value
    region = _g(d, "Geographic_Region")
    if region == "Arid/Desert":
        region = "Subtropical"
    if region == "Mediterranean":
        region = "Temperate"
    if region == "Polar/Cold":
        region = "Polar/Cold"
    for r in ["Polar/Cold", "Subtropical", "Temperate", "Tropical"]:
        row[f"Geographic_Region_{r}"] = 1 if region == r else 0

    # ── One-hot: Diet_Type ──────────────────────────────────────────────────
    diet = _g(d, "Diet_Type")
    for dt in ["Omnivore", "Pescatarian", "Vegan", "Vegetarian"]:
        row[f"Diet_Type_{dt}"] = 1 if diet == dt else 0

    # ── One-hot: Smoking_Status ─────────────────────────────────────────────
    smoke = _g(d, "Smoking_Status")
    for s in ["Current", "Former", "Never"]:
        row[f"Smoking_Status_{s}"] = 1 if smoke == s else 0

    # ── One-hot: Alcohol_Consumption ────────────────────────────────────────
    alc = _g(d, "Alcohol_Consumption")
    for a in ["Heavy", "Moderate", "None", "Occasional"]:
        row[f"Alcohol_Consumption_{a}"] = 1 if alc == a else 0

    # ── One-hot: Chronic_Condition ──────────────────────────────────────────
    cc = _g(d, "Chronic_Condition")
    if not cc or cc.strip() == "":
        cc = "None"
    for c in ["CKD", "Celiac", "Diabetes", "Hypertension", "Hypothyroid", "IBD", "None"]:
        row[f"Chronic_Condition_{c}"] = 1 if cc == c else 0

    # ── One-hot: Medication_Use ─────────────────────────────────────────────
    med = _g(d, "Medication_Use")
    if not med or med.strip() == "":
        med = "None"
    for m in ["Antacids/PPIs", "Diuretics", "Metformin", "None", "OCP", "Steroids"]:
        row[f"Medication_Use_{m}"] = 1 if med == m else 0

    # ── One-hot: Pregnancy_Status ───────────────────────────────────────────
    preg = _g(d, "Pregnancy_Status")
    row["Pregnancy_Status_No"]   = 1 if preg in ("No", "False", False)   else 0
    row["Pregnancy_Status_None"] = 1 if preg in ("", "None", None, "N/A") else 0
    row["Pregnancy_Status_Yes"]  = 1 if preg in ("Yes", "True", True)    else 0
    # If none matched, default to None
    if not any([row["Pregnancy_Status_No"], row["Pregnancy_Status_None"], row["Pregnancy_Status_Yes"]]):
        row["Pregnancy_Status_None"] = 1

    return pd.DataFrame([row])[_FEATURE_NAMES]


def _run_model(d: dict) -> tuple:
    if _model is None or _FEATURE_NAMES is None:
        return _fallback(d), 0.70

    try:
        df   = _build_features(d)
        pred = _model.predict(df)[0]
        conf = float(_model.predict_proba(df).max())
        # Use model's own internal class list — avoids label encoder mismatch
        # ✅ FIX: pred is already the class label (string after retraining)
        # AFTER
        DEFICIENCY_NAMES = {0: "Calcium", 1: "Folate", 2: "Iron", 3: "Magnesium", 4: "Vitamin B12", 5: "Vitamin D",
                            6: "Zinc"}
        lbl = DEFICIENCY_NAMES.get(int(pred), f"Unknown ({pred})")
        return lbl, conf
    except Exception as e:
        print(f"Prediction error: {e}")
        return _fallback(d), 0.65


def _fallback(d: dict) -> str:
    def s(k): return _SYM_MAP.get(d.get(k, "No"), 0)
    if s("Fatigue") + s("Pallor") + s("Shortness_of_Breath") >= 3:
        return "Iron"
    if s("Bone_Pain") >= 1 and _SUN_MAP.get(d.get("Sun_Exposure", "Moderate"), 1) == 0:
        return "Vitamin D"
    if s("Tingling_Numbness") + s("Memory_Cognitive_Issues") >= 2:
        return "Vitamin B12"
    if s("Muscle_Cramps") + s("Anxiety_and_Poor_Sleep") >= 2:
        return "Magnesium"
    return "Vitamin D"

#Hosting
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))