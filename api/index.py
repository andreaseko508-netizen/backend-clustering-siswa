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
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

app = FastAPI(title="SIMORBATAS Python AI Runtime (Vercel)", version="1.5.0")

# Initialize Firebase Admin SDK
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
            cred_path = os.path.join(base_dir, "serviceAccountKey.json")
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                print("Firebase initialized via serviceAccountKey.json.")

    if firebase_admin._apps:
        db = firestore.client()
except Exception as e:
    print(f"Error initializing Firebase: {e}")

sessions: Dict[str, Dict[str, Any]] = {}

def sync_session_to_firebase(session_id: str):
    if not db or session_id not in sessions: return
    try:
        session = sessions[session_id].copy()
        if "df" in session and isinstance(session["df"], pd.DataFrame):
            df_cleaned = session["df"].replace([np.inf, -np.inf], np.nan).fillna(0)
            session["df_records"] = df_cleaned.to_dict(orient="records")
            del session["df"]
        db.collection("python_sessions").document(session_id).set(session)
    except Exception as e: print(f"Failed to sync: {e}")

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
        if step_name not in checklist: checklist.append(step_name)
        sessions[x_session_id]["audit"]["execution_checklist"] = checklist

def calculate_cluster_metrics(df, features, assignments, k):
    try:
        X = df[features].select_dtypes(include=[np.number]).fillna(0)
        unique_labels = np.unique(assignments)
        dbi = float(davies_bouldin_score(X, assignments)) if len(unique_labels) > 1 else 0.0
        sil = float(silhouette_score(X, assignments)) if len(unique_labels) > 1 else 0.0
        dist = {str(i): {"count": int(np.sum(assignments == i)), "percentage": float(np.sum(assignments == i) / len(df) * 100)} for i in range(k)}
        profiles = {str(i): df[assignments == i][features].mean(numeric_only=True).to_dict() for i in range(k)}
        return {"davies_bouldin_index": dbi, "silhouette_score": sil, "distribution": dist, "cluster_profiles": profiles, "dbi": dbi}
    except: return {"davies_bouldin_index": 0.0, "silhouette_score": 0.0, "distribution": {}, "cluster_profiles": {}, "dbi": 0.0}

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "Online", "engine": "SIMORBATAS-Vercel", "firebase": "Connected" if db else "Offline"}

@app.get("/health")
async def health():
    return {"status": "UP", "firebase": "Connected" if db else "Offline"}

@app.post("/stepwise/upload/")
async def stepwise_upload(file: UploadFile = File(...), x_session_id: Optional[str] = Header(None)):
    if not x_session_id: x_session_id = str(uuid.uuid4())
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(content))
        sessions[x_session_id] = {"df": df, "filename": file.filename, "config": {}, "metrics": {}, "checkpoints": {"Data Asli": df.head(100).to_dict(orient="records")}, "audit": {"initial_rows": len(df), "initial_cols": len(df.columns), "missing_before": int(df.isnull().sum().sum()), "outliers_removed": 0, "normalization_method": "None", "execution_checklist": []}}
        sync_session_to_firebase(x_session_id)
        return {"status": "success", "jumlah_data": len(df), "columns": list(df.columns), "session_id": x_session_id}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/stepwise/raw-data/")
async def get_raw_data(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    return {"columns": list(df.columns), "total_rows": int(len(df)), "data": df.replace([np.inf, -np.inf], np.nan).fillna(0).head(100).to_dict(orient="records")}

@app.post("/stepwise/cleaning/")
async def stepwise_cleaning(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    initial_rows = len(df)
    sessions[x_session_id]["checkpoints"]["Pembersihan Data (Sebelum)"] = df.head(100).to_dict(orient="records")
    df = df.dropna(how='all').dropna(axis=1, how='all').drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns: df[col] = df[col].astype(str).str.strip()
    sessions[x_session_id]["df"] = df
    sessions[x_session_id]["checkpoints"]["Pembersihan Data (Sesudah)"] = df.head(100).to_dict(orient="records")
    add_to_checklist(x_session_id, "Cleaning")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "final_rows": len(df), "log": f"Cleaning selesai."}

@app.post("/stepwise/missing-value/")
async def stepwise_missing(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    sessions[x_session_id]["checkpoints"]["Imputasi Nilai Kosong (Sebelum)"] = df.head(100).to_dict(orient="records")
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols: df[col] = df[col].fillna(df[col].median())
    sessions[x_session_id]["df"] = df
    sessions[x_session_id]["checkpoints"]["Imputasi Nilai Kosong (Sesudah)"] = df.head(100).to_dict(orient="records")
    add_to_checklist(x_session_id, "Missing Value")
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.get("/stepwise/missing-scan")
async def missing_scan(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    num_cols = df.select_dtypes(include=['number']).columns
    missing_stats = {col: {"count": int(df[col].isnull().sum()), "median": float(df[col].median())} for col in num_cols if df[col].isnull().sum() > 0}
    return {"status": "success", "total_missing": int(df.isnull().sum().sum()), "missing_by_column": missing_stats}

@app.post("/stepwise/outlier-detection/")
async def stepwise_outlier(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=['number']).columns))
    num_df = df[features].select_dtypes(include=['number'])
    Q1, Q3 = num_df.quantile(0.25), num_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers_mask = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)
    sessions[x_session_id]["checkpoints"]["Deteksi Outlier (Sesudah)"] = df[~outliers_mask].head(100).to_dict(orient="records")
    add_to_checklist(x_session_id, "Outlier")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "outlier_count": int(outliers_mask.sum())}

@app.post("/stepwise/conversion/")
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
    sessions[x_session_id]["checkpoints"]["Konversi Kategorikal (Sesudah)"] = df.head(100).to_dict(orient="records")
    add_to_checklist(x_session_id, "Conversion")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "mappings": mapping_details}

@app.get("/stepwise/normalization-stats/")
async def get_norm_stats(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    num_df = df.select_dtypes(include=['number'])
    stats = {col: {"min": float(num_df[col].min()), "max": float(num_df[col].max()), "mean": float(num_df[col].mean())} for col in num_df.columns}
    return {"status": "success", "stats": stats}

@app.post("/stepwise/normalization/")
async def stepwise_norm(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    from sklearn.preprocessing import MinMaxScaler
    df = sessions[x_session_id]["df"]
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
        sessions[x_session_id]["df"] = df
        sessions[x_session_id]["checkpoints"]["Normalisasi Min-Max (Sesudah)"] = df.head(100).to_dict(orient="records")
        add_to_checklist(x_session_id, "Normalization")
        sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/stepwise/standardization/")
async def stepwise_standard(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    from sklearn.preprocessing import StandardScaler
    df = sessions[x_session_id]["df"]
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        df[num_cols] = StandardScaler().fit_transform(df[num_cols])
        sessions[x_session_id]["df"] = df
        sessions[x_session_id]["checkpoints"]["Standardisasi Z-Score (Sesudah)"] = df.head(100).to_dict(orient="records")
        add_to_checklist(x_session_id, "Standardization")
        sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.get("/stepwise/quality-report/")
async def get_quality_report(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[x_session_id]
    df = session["df"]
    num_cols = list(df.select_dtypes(include=['number']).columns)
    return {"status": "success", "rows": len(df), "cols": len(df.columns), "numeric_features": len(num_cols), "completeness": 1.0 - (df.isnull().sum().sum() / df.size if df.size > 0 else 0), "is_suitable": len(df) > 0 and len(num_cols) >= 2, "execution_checklist": session["audit"].get("execution_checklist", [])}

@app.get("/stepwise/checkpoints/")
async def get_checkpoints(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "checkpoints": sessions[x_session_id].get("checkpoints", {})}

@app.get("/stepwise/universal-dataset/")
async def get_universal_dataset(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    return {"columns": list(sessions[x_session_id]["df"].columns), "data": sessions[x_session_id]["df"].head(500).to_dict(orient="records")}

@app.get("/stepwise/session-state/")
async def get_session_state(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    return {"state": "UPLOADED" if x_session_id in sessions else "IDLE"}

@app.post("/stepwise/elbow/")
async def stepwise_elbow(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    from sklearn.cluster import KMeans
    X = sessions[x_session_id]["df"].select_dtypes(include=[np.number]).fillna(0)
    wcss = [{"k": i, "wcss": float(KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42).fit(X).inertia_)} for i in range(1, 11)]
    return {"status": "success", "data": wcss}

@app.post("/stepwise/init-centroids/")
async def init_centroids_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3, "init_method": "random"})):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df, k = sessions[x_session_id]["df"], params.get("k", 3)
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=[np.number]).columns))
    num_df = df[features].select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
    centroids = num_df.sample(n=k).values.tolist()
    sessions[x_session_id]["algo_state"] = {"iteration": 0, "centroids": centroids, "features": features, "k": k, "history": [], "is_converged": False}
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "centroids": centroids, "features": features, "message": "Inisialisasi berhasil."}

@app.post("/stepwise/calculate-distances/")
async def calculate_distances_step(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    if not state: raise HTTPException(status_code=400, detail="Algo state missing")
    num_df = sessions[x_session_id]["df"][state["features"]].select_dtypes(include=[np.number]).fillna(0)
    centroids = np.array(state["centroids"])
    distances = [np.linalg.norm(centroids - row.values, axis=1).tolist() for _, row in num_df.iterrows()]
    state["distances"] = distances
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "distance_matrix_sample": distances[:5], "sample_work": {"distances": distances[0]}}

@app.post("/stepwise/assign-clusters/")
async def assign_clusters_step(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    if not state or "distances" not in state: raise HTTPException(status_code=400, detail="Distances not calculated")
    distances = np.array(state["distances"])
    assignments = np.argmin(distances, axis=1).tolist()
    state["assignments"] = assignments
    state["current_wcss"] = float(np.sum(np.min(distances, axis=1)**2))
    counts = {str(i): int(np.sum(np.array(assignments) == i)) for i in range(state["k"])}
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "assignments": assignments, "wcss": state["current_wcss"], "counts": counts}

@app.post("/stepwise/update-centroids/")
async def update_centroids_step(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    num_df = sessions[x_session_id]["df"][state["features"]].fillna(0)
    assignments = np.array(state["assignments"])
    old_centroids = np.array(state["centroids"])
    new_centroids = []
    for i in range(state["k"]):
        cluster_points = num_df[assignments == i]
        new_centroids.append(cluster_points.mean(axis=0).values.tolist() if len(cluster_points) > 0 else old_centroids[i].tolist())

    movement = float(np.linalg.norm(np.array(new_centroids) - old_centroids))
    state["centroids"] = new_centroids
    state["iteration"] += 1
    state["history"].append({"iter": state["iteration"], "wcss": state.get("current_wcss", 0.0), "movement": movement})
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "new_centroids": new_centroids, "iteration": state["iteration"], "movement": movement, "sample_work": {"explanation": "Centroid baru dihitung dari rata-rata anggota cluster."}}

@app.post("/stepwise/check-convergence/")
async def check_convergence(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    if not state: raise HTTPException(status_code=400, detail="Algo state missing")
    is_converged = state["history"][-1]["movement"] < 1e-4 if state["history"] else False
    state["is_converged"] = is_converged
    evaluation = calculate_cluster_metrics(sessions[x_session_id]["df"], state["features"], np.array(state["assignments"]), state["k"]) if is_converged else {}
    if is_converged:
        sessions[x_session_id]["df"]["cluster"] = state["assignments"]
        sessions[x_session_id]["metrics"] = evaluation
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "is_converged": is_converged, "iteration": state["iteration"], "history": state["history"], "centroids": state["centroids"], "evaluation": evaluation}

@app.post("/stepwise/auto-converge/")
async def auto_converge(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    df = sessions[x_session_id]["df"]
    features = state["features"]
    X = df[features].fillna(0).values

    # MANUAL K-MEANS LOOP to capture real history for research validation
    # This prevents the 'all 0.0' issue in Step 18
    centroids = np.array(state["centroids"])
    history = []
    assignments = np.zeros(len(X))

    for i in range(1, 101): # Max 100 iterations
        # 1. Calculate Distances & Assignments
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        assignments = np.argmin(dists, axis=1)
        wcss = float(np.sum(np.min(dists, axis=1)**2))

        # 2. Update Centroids
        new_centroids = np.array([X[assignments == j].mean(axis=0) if len(X[assignments == j]) > 0 else centroids[j] for j in range(state["k"])])
        movement = float(np.linalg.norm(new_centroids - centroids))

        # 3. Record History
        history.append({"iter": i, "wcss": wcss, "movement": movement})

        centroids = new_centroids
        if movement < 1e-4: break

    state.update({"iteration": len(history), "centroids": centroids.tolist(), "assignments": assignments.tolist(), "is_converged": True, "history": history})

    evaluation = calculate_cluster_metrics(df, features, assignments, state["k"])
    df["cluster"] = assignments.tolist()
    sessions[x_session_id].update({"df": df, "metrics": evaluation})

    sync_session_to_firebase(x_session_id)
    return {"status": "success", "is_converged": True, "iteration": state["iteration"], "history": history, "centroids": state["centroids"], "evaluation": evaluation}

@app.post("/stepwise/run-kmeans/")
async def run_kmeans_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    await ensure_session(x_session_id)
    from sklearn.cluster import KMeans
    df = sessions[x_session_id]["df"]
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=[np.number]).columns))
    model = KMeans(n_clusters=params.get("k", 3), n_init=10, random_state=42).fit(df[features].fillna(0))
    df['cluster'] = model.labels_
    metrics = calculate_cluster_metrics(df, features, model.labels_, params.get("k", 3))
    metrics.update({"wcss": model.inertia_, "iterations": model.n_iter_, "centroids": model.cluster_centers_.tolist(), "feature_names": features})
    sessions[x_session_id].update({"df": df, "metrics": metrics})
    sync_session_to_firebase(x_session_id)
    return {"status": "SUCCESS", "metrics": metrics}

@app.get("/stepwise/final-analysis/")
async def get_final_analysis(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    session = sessions[x_session_id]
    metrics = session.get("metrics", {})
    return {"status": "success", "jumlah_data": len(session["df"]), "metrics": metrics, "silhouette_score": metrics.get("silhouette_score", 0.0), "davies_bouldin_index": metrics.get("davies_bouldin_index", 0.0), "wcss": metrics.get("wcss", 0.0), "iterations": metrics.get("iterations", 0), "cluster_distribution": metrics.get("distribution", {}), "cluster_profiles": metrics.get("cluster_profiles", {}), "centroids": metrics.get("centroids", []), "feature_names": metrics.get("feature_names", []), "hasil_cluster": session["df"].to_dict(orient="records")}

@app.post("/stepwise/save_config/")
@app.post("/stepwise/mapping-config/")
async def stepwise_mapping(x_session_id: Optional[str] = Header(None), config: Dict[str, Any] = Body(...)):
    await ensure_session(x_session_id)
    sessions[x_session_id]["config"].update(config)
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.get("/stepwise/export-excel/")
async def export_excel(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    session = sessions[x_session_id]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, data in session.get("checkpoints", {}).items():
            if data: pd.DataFrame(data).to_excel(writer, sheet_name=name[:31], index=False)
        session["df"].to_excel(writer, sheet_name="Hasil Akhir", index=False)
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=Riset_{x_session_id[:8]}.xlsx"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
