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

# Global Seed
np.random.seed(42)

# Session Storage
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
    print(f"Error initializing Firebase: {e}")

def sync_session_to_firebase(session_id: str):
    if not db or session_id not in sessions: return
    try:
        session = sessions[session_id].copy()

        # Serialize Scaler if exists
        if "scaler" in session and session["scaler"] is not None:
            try:
                session["scaler_b64"] = base64.b64encode(pickle.dumps(session["scaler"])).decode('utf-8')
                del session["scaler"]
            except Exception as e: print(f"Scaler serialization failed: {e}")

        if "df" in session and isinstance(session["df"], pd.DataFrame):
            df_cleaned = session["df"].replace([np.inf, -np.inf], np.nan).fillna(0)
            session["df_records"] = df_cleaned.to_dict(orient="records")
            del session["df"]

        # Ensure all_results is serializable
        if "all_results" in session:
            serialized_results = {}
            for k, v in session["all_results"].items():
                if "df" in v: del v["df"]
                serialized_results[k] = v
            session["all_results"] = serialized_results

        db.collection("python_sessions").document(session_id).set(session)
    except Exception as e: print(f"Failed to sync: {e}")

async def ensure_session(x_session_id: str):
    if not x_session_id: return
    if x_session_id not in sessions:
        if db:
            doc = db.collection("python_sessions").document(x_session_id).get()
            if doc.exists:
                data = doc.to_dict()
                if "scaler_b64" in data and data["scaler_b64"]:
                    try:
                        data["scaler"] = pickle.loads(base64.b64decode(data["scaler_b64"]))
                    except: pass
                    del data["scaler_b64"]
                if "df_records" in data:
                    data["df"] = pd.DataFrame(data["df_records"])
                    del data["df_records"]
                sessions[x_session_id] = data

def add_to_checklist(x_session_id: str, step_name: str):
    if x_session_id in sessions:
        if "audit" not in sessions[x_session_id]:
            sessions[x_session_id]["audit"] = {"execution_checklist": []}
        checklist = sessions[x_session_id]["audit"].get("execution_checklist", [])
        if step_name not in checklist:
            checklist.append(step_name)
        sessions[x_session_id]["audit"]["execution_checklist"] = checklist
        sync_session_to_firebase(x_session_id)

def get_representative_data(df):
    if len(df) <= 5:
        return df.to_dict(orient="records")

    first_three = df.head(3)
    last_two = df.tail(2)

    representative = pd.concat([first_three, last_two])
    return representative.to_dict(orient="records")
