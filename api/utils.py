import os
import json
import pickle
import base64
import firebase_admin
import pandas as pd
import numpy as np
from firebase_admin import credentials, firestore
from typing import Optional, List, Dict, Any
from fastapi import HTTPException

# Global Seed for Determinism
np.random.seed(42)

# Session Storage (RAM)
sessions: Dict[str, Dict[str, Any]] = {}
audit_checkpoints: Dict[str, Dict[str, pd.DataFrame]] = {}

# Initialize Firebase
db = None
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    if not firebase_admin._apps:
        firebase_creds_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if firebase_creds_json:
            creds_dict = json.loads(firebase_creds_json)
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        else:
            cred_path = os.path.join(base_dir, "serviceAccountKey.json")
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)

    if firebase_admin._apps:
        db = firestore.client()
except Exception as e:
    print(f"Firebase Init Fail: {e}")

def safe_float(val):
    """S2 HARDENING: Ensures numeric values are JSON-safe (No NaN/Inf)."""
    try:
        if val is None or np.isnan(val) or np.isinf(val): return 0.0
        return float(val)
    except: return 0.0

def ensure_audit(x_session_id: str):
    """Ensures audit dictionary exists for the session."""
    if x_session_id not in audit_checkpoints:
        audit_checkpoints[x_session_id] = {}

def sync_session_to_firebase(session_id: str):
    """Backs up the current session state to Firestore for serverless persistence."""
    if not db or session_id not in sessions: return
    try:
        raw_session = sessions[session_id]
        session = {}
        for k, v in raw_session.items():
            if k in ["df", "unscaled_df"]:
                continue
            if k == "scaler":
                if v is not None:
                    try:
                        session["scaler_b64"] = base64.b64encode(pickle.dumps(v)).decode('utf-8')
                    except Exception: pass
                continue
            session[k] = v

        if "df" in raw_session and isinstance(raw_session["df"], pd.DataFrame):
            df = raw_session["df"]
            session["df_columns"] = [str(c) for c in df.columns]
            df_safe = df.copy()
            for col in df_safe.columns:
                if pd.api.types.is_datetime64_any_dtype(df_safe[col]):
                    df_safe[col] = df_safe[col].astype(str)
            df_safe = df_safe.replace([np.inf, -np.inf], np.nan).fillna(0)
            session["df_records"] = df_safe.to_dict(orient="records")

        db.collection("python_sessions").document(session_id).set(session)
    except Exception as e:
        print(f"Sync fail for {session_id}: {e}")

async def ensure_session(x_session_id: str):
    """Ensures the session is loaded into RAM, restoring from Firestore if necessary."""
    if not x_session_id: return

    # If not in RAM, try to pull from Firestore
    if x_session_id not in sessions:
        if db:
            doc = db.collection("python_sessions").document(x_session_id).get()
            if doc.exists:
                data = doc.to_dict()

                # Restore Scaler
                if "scaler_b64" in data and data["scaler_b64"]:
                    try:
                        data["scaler"] = pickle.loads(base64.b64decode(data["scaler_b64"]))
                    except: pass

                # Restore DataFrame
                if "df_records" in data:
                    cols = data.get("df_columns")
                    df = pd.DataFrame(data["df_records"])
                    if cols: df = df.reindex(columns=cols)
                    for col in df.columns:
                        try: df[col] = pd.to_numeric(df[col], errors='ignore')
                        except: pass
                    data["df"] = df

                sessions[x_session_id] = data

def add_to_checklist(x_session_id: str, step_name: str):
    """Updates the research audit trail."""
    if x_session_id in sessions:
        if "audit" not in sessions[x_session_id]:
            sessions[x_session_id]["audit"] = {"execution_checklist": []}
        checklist = sessions[x_session_id]["audit"].get("execution_checklist", [])
        if step_name not in checklist:
            checklist.append(step_name)
        sessions[x_session_id]["audit"]["execution_checklist"] = checklist
        sync_session_to_firebase(x_session_id)

def clean_df_for_json(df):
    """
    Converts DataFrame records to 100% JSON-safe Python list of dicts.
    Replaces NaN, Inf, -Inf, NaT, and Timestamps with safe strings/None/0.
    Prevents 'ValueError: Out of range float values are not JSON compliant: nan'!
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    records = []
    for _, row in df_clean.iterrows():
        row_dict = {}
        for col, val in row.items():
            col_str = str(col).strip() if col is not None else "Col"
            if pd.isna(val) or val is None or str(val).lower() in ['nan', 'none', 'nat']:
                row_dict[col_str] = ""
            elif isinstance(val, (pd.Timestamp, pd.Timedelta)):
                row_dict[col_str] = str(val)
            elif isinstance(val, (float, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    row_dict[col_str] = ""
                else:
                    row_dict[col_str] = float(val)
            elif isinstance(val, (int, np.integer)):
                row_dict[col_str] = int(val)
            else:
                row_dict[col_str] = str(val)
        records.append(row_dict)

    return records

def get_representative_data(df):
    """Returns a visual preview for UI efficiency."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty: return []
    if len(df) <= 5: return clean_df_for_json(df)
    return clean_df_for_json(pd.concat([df.head(3), df.tail(2)]))
