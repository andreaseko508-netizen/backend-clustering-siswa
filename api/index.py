from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Body
from fastapi.responses import StreamingResponse
import importlib
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import io
import uuid
import json
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Optional, List, Dict, Any

# VERCEL COMPATIBILITY: Ensure the current directory and parent are in sys.path
# This allows 'import sdk...' and 'import plugins...' to work correctly.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Import SDK models after path adjustment
try:
    from sdk.models import ExecutionContext, ExecutionResult
except ImportError:
    # Fallback/Dummy for initial build check
    class ExecutionContext: pass
    class ExecutionResult: pass

app = FastAPI(title="SIMORBATAS Python AI Runtime (Vercel)", version="1.1.0")

# Initialize Firebase Admin SDK with Singleton Pattern to prevent Vercel Crash
db = None
try:
    if not firebase_admin._apps:
        firebase_creds_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if firebase_creds_json:
            creds_dict = json.loads(firebase_creds_json)
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
            print("Firebase initialized via Environment Variable.")
        else:
            # Local development fallback
            cred_path = os.path.join(base_dir, "serviceAccountKey.json")
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                print("Firebase initialized via serviceAccountKey.json.")

    if firebase_admin._apps:
        db = firestore.client()
except Exception as e:
    print(f"Error initializing Firebase: {e}")

# In-memory session storage (Transient but backed by Firebase)
sessions: Dict[str, Dict[str, Any]] = {}

def sync_session_to_firebase(session_id: str):
    if not db or session_id not in sessions:
        return
    try:
        session = sessions[session_id].copy()
        if "df" in session and isinstance(session["df"], pd.DataFrame):
            df_cleaned = session["df"].replace([np.inf, -np.inf], np.nan).fillna(0)
            session["df_records"] = df_cleaned.to_dict(orient="records")
            del session["df"]
        db.collection("python_sessions").document(session_id).set(session)
    except Exception as e:
        print(f"Failed to sync session {session_id}: {e}")

async def ensure_session(x_session_id: str):
    if not x_session_id: return
    if x_session_id not in sessions:
        if db:
            doc = db.collection("python_sessions").document(x_session_id).get()
            if doc.exists:
                data = doc.to_dict()
                if "df_records" in data:
                    data["df"] = pd.DataFrame(data["df_records"])
                    del data["df_records"]
                sessions[x_session_id] = data

def add_to_checklist(x_session_id: str, step_name: str):
    if x_session_id in sessions:
        checklist = sessions[x_session_id]["audit"].get("execution_checklist", [])
        if step_name not in checklist:
            checklist.append(step_name)
        sessions[x_session_id]["audit"]["execution_checklist"] = checklist

def calculate_cluster_metrics(df, features, assignments, k):
    try:
        X = df[features].select_dtypes(include=[np.number]).fillna(0)
        unique_labels = np.unique(assignments)
        dbi = float(davies_bouldin_score(X, assignments)) if len(unique_labels) > 1 else 0.0
        sil = float(silhouette_score(X, assignments)) if len(unique_labels) > 1 else 0.0
        dist = {str(i): {"count": int(np.sum(assignments == i)), "percentage": float(np.sum(assignments == i) / len(df) * 100)} for i in range(k)}
        profiles = {str(i): df[assignments == i][features].mean(numeric_only=True).to_dict() for i in range(k)}
        return {"davies_bouldin_index": dbi, "silhouette_score": sil, "distribution": dist, "cluster_profiles": profiles}
    except:
        return {"davies_bouldin_index": 0.0, "silhouette_score": 0.0, "distribution": {}, "cluster_profiles": {}}

# --- ENDPOINTS ---

@app.get("/")
@app.get("/api")
async def root():
    return {
        "status": "Online",
        "engine": "SIMORBATAS-Vercel",
        "message": "Server Riset Clustering Siswa siap melayani aplikasi Android Anda.",
        "endpoints": ["/api/health", "/api/stepwise/upload", "/api/stepwise/final-analysis"]
    }

@app.get("/api/health")
async def health():
    return {"status": "UP", "engine": "Vercel Serverless Python", "firebase": "Connected" if db else "Offline"}

@app.post("/api/stepwise/upload/")
async def stepwise_upload(file: UploadFile = File(...), x_session_id: Optional[str] = Header(None)):
    if not x_session_id: x_session_id = str(uuid.uuid4())
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(content))
        sessions[x_session_id] = {
            "df": df,
            "filename": file.filename,
            "config": {},
            "metrics": {},
            "checkpoints": {"Data Asli": df.head(100).to_dict(orient="records")},
            "audit": {"initial_rows": len(df), "initial_cols": len(df.columns), "missing_before": int(df.isnull().sum().sum()), "outliers_removed": 0, "normalization_method": "None", "execution_checklist": []}
        }
        sync_session_to_firebase(x_session_id)
        return {"status": "success", "jumlah_data": len(df), "columns": list(df.columns), "session_id": x_session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stepwise/raw-data/")
async def get_raw_data(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    return {"columns": list(df.columns), "total_rows": int(len(df)), "data": df.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records")}

@app.post("/api/stepwise/cleaning/")
async def stepwise_cleaning(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    initial_rows = len(df)
    sessions[x_session_id]["checkpoints"]["Pembersihan Data (Sebelum)"] = df.to_dict(orient="records")
    df = df.dropna(how='all').dropna(axis=1, how='all').drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns: df[col] = df[col].astype(str).str.strip()
    sessions[x_session_id]["df"] = df
    sessions[x_session_id]["checkpoints"]["Pembersihan Data (Sesudah)"] = df.to_dict(orient="records")
    add_to_checklist(x_session_id, "Cleaning")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "final_rows": len(df), "log": f"Cleaning selesai: {initial_rows - len(df)} baris dihapus."}

@app.post("/api/stepwise/missing-value/")
async def stepwise_missing(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    sessions[x_session_id]["checkpoints"]["Imputasi Nilai Kosong (Sebelum)"] = df.to_dict(orient="records")
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols: df[col] = df[col].fillna(df[col].median())
    sessions[x_session_id]["df"] = df
    sessions[x_session_id]["checkpoints"]["Imputasi Nilai Kosong (Sesudah)"] = df.to_dict(orient="records")
    add_to_checklist(x_session_id, "Missing Value")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "log": "Imputasi selesai."}

@app.post("/api/stepwise/conversion/")
async def stepwise_conversion(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    features = sessions[x_session_id]["config"].get("features", [])
    cat_cols = df[features].select_dtypes(include=['object', 'category']).columns
    mapping_details = {}
    for col in cat_cols:
        codes, uniques = pd.factorize(df[col])
        df[col] = codes
        mapping_details[col] = {str(i): str(val) for i, val in enumerate(uniques)}
    sessions[x_session_id]["df"] = df
    sessions[x_session_id]["checkpoints"]["Konversi Kategorikal (Sesudah)"] = df.to_dict(orient="records")
    add_to_checklist(x_session_id, "Conversion")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "mappings": mapping_details}

@app.post("/api/stepwise/normalization/")
async def stepwise_norm(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    from sklearn.preprocessing import MinMaxScaler
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        sessions[x_session_id]["df"] = df
        sessions[x_session_id]["checkpoints"]["Normalisasi Min-Max (Sesudah)"] = df.to_dict(orient="records")
        add_to_checklist(x_session_id, "Normalization")
        sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/api/stepwise/save_config/")
@app.post("/api/stepwise/mapping-config/")
async def stepwise_mapping(x_session_id: Optional[str] = Header(None), config: Dict[str, Any] = Body(...)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    sessions[x_session_id]["config"].update(config)
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/api/stepwise/run-kmeans/")
async def run_kmeans_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    from sklearn.cluster import KMeans
    df = sessions[x_session_id]["df"]
    features = sessions[x_session_id]["config"].get("features", [])
    if not features: features = list(df.select_dtypes(include=[np.number]).columns)
    X = df[features].select_dtypes(include=[np.number]).fillna(0)
    k = params.get("k", 3)
    model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    clusters = model.fit_predict(X)
    df['cluster'] = clusters
    for i in range(k):
        df[f"dist_c{i}"] = np.linalg.norm(X.values - model.cluster_centers_[i], axis=1)

    metrics = calculate_cluster_metrics(df, features, clusters, k)
    metrics["wcss"] = float(model.inertia_)
    metrics["iterations"] = int(model.n_iter_)
    metrics["centroids"] = model.cluster_centers_.tolist()
    metrics["feature_names"] = list(features)

    sessions[x_session_id]["df"] = df
    sessions[x_session_id]["metrics"] = metrics
    sessions[x_session_id]["checkpoints"]["Hasil Akhir"] = df.head(100).to_dict(orient="records")
    sync_session_to_firebase(x_session_id)
    return {"status": "SUCCESS", "metrics": metrics}

@app.get("/api/stepwise/final-analysis/")
async def get_final_analysis(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[x_session_id]
    metrics = session.get("metrics", {})
    return {
        "status": "success",
        "jumlah_data": len(session["df"]),
        "metrics": metrics,
        "silhouette_score": metrics.get("silhouette_score", 0.0),
        "davies_bouldin_index": metrics.get("davies_bouldin_index", 0.0),
        "wcss": metrics.get("wcss", 0.0),
        "iterations": metrics.get("iterations", 0),
        "cluster_distribution": metrics.get("distribution", {}),
        "cluster_profiles": metrics.get("cluster_profiles", {}),
        "centroids": metrics.get("centroids", []),
        "feature_names": metrics.get("feature_names", []),
        "hasil_cluster": session["df"].to_dict(orient="records")
    }

@app.get("/api/stepwise/checkpoints/")
async def get_checkpoints(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "checkpoints": sessions[x_session_id].get("checkpoints", {})}

@app.get("/api/stepwise/export-excel/")
async def export_excel(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[x_session_id]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        checkpoints = session.get("checkpoints", {})
        for name, data in checkpoints.items():
             if data: pd.DataFrame(data).to_excel(writer, sheet_name=name[:30], index=False)
        if "df" in session: session["df"].to_excel(writer, sheet_name="Hasil Akhir Final", index=False)
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=Laporan_Riset_{x_session_id[:8]}.xlsx"})

# Vercel entry point
# No if __name__ == "__main__" needed for serverless, but kept for compatibility
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
