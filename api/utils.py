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
        # Create a clean copy for serialization
        session = sessions[session_id].copy()

        # 1. Serialize Scaler
        if "scaler" in session and session["scaler"] is not None:
            try:
                session["scaler_b64"] = base64.b64encode(pickle.dumps(session["scaler"])).decode('utf-8')
                del session["scaler"]
            except: pass

        # 2. Serialize DataFrame
        if "df" in session and isinstance(session["df"], pd.DataFrame):
            df = session["df"]
            session["df_columns"] = [str(c) for c in df.columns]
            df_safe = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            session["df_records"] = df_safe.to_dict(orient="records")
            del session["df"]

        # 3. Write to Firestore (Non-blocking as possible)
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

def get_representative_data(df):
    """Returns a visual preview for UI efficiency."""
    if df is None or df.empty: return []
    if len(df) <= 5: return df.to_dict(orient="records")
    return pd.concat([df.head(3), df.tail(2)]).to_dict(orient="records")
