from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Body
from fastapi.responses import StreamingResponse
import importlib
import os
import sys
import pandas as pd
import numpy as np
import time
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score, adjusted_rand_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import chi2, shapiro
import io
import shap
import uuid
import json
import pickle
import base64
import firebase_admin
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from fpdf import FPDF
from firebase_admin import credentials, firestore
from typing import Optional, List, Dict, Any
import google.generativeai as genai

# S2 AUDIT: Global Seed for Scientific Determinism
np.random.seed(42)

# Ensure the current directory and parent are in sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

app = FastAPI(title="SIMORBATAS Python AI Runtime (Local Server)", version="1.7.5")

# Initialize Firebase Admin SDK
db = None
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

class ResearchReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'SIMORBATAS: Laporan Hasil Riset Pengelompokan Siswa', border=False, align='C')
        self.ln(15)
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}/{{nb}} - Digital Signature: {str(uuid.uuid4())[:8]}', align='C')
    def chapter_title(self, label):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(226, 232, 240)
        self.cell(0, 10, label, border=True, ln=True, fill=True)
        self.ln(4)
    def chapter_body(self, body):
        self.set_font('helvetica', '', 10)
        self.multi_cell(0, 7, body)
        self.ln()
    def add_table(self, header, data):
        self.set_font('helvetica', 'B', 10)
        col_width = self.epw / len(header)
        for h in header: self.cell(col_width, 7, h, border=1, align='C')
        self.ln()
        self.set_font('helvetica', '', 9)
        for row in data:
            for item in row: self.cell(col_width, 7, str(item), border=1, align='C')
            self.ln()
        self.ln(5)

sessions: Dict[str, Dict[str, Any]] = {}

def sync_session_to_firebase(session_id: str):
    if not db or session_id not in sessions: return
    try:
        session = sessions[session_id].copy()
        if "scaler" in session and session["scaler"] is not None:
            try:
                session["scaler_b64"] = base64.b64encode(pickle.dumps(session["scaler"])).decode('utf-8')
                del session["scaler"]
            except: pass
        if "df" in session and isinstance(session["df"], pd.DataFrame):
            df_cleaned = session["df"].replace([np.inf, -np.inf], np.nan).fillna(0)
            session["df_records"] = df_cleaned.to_dict(orient="records")
            del session["df"]
        db.collection("python_sessions").document(session_id).set(session)
    except Exception as e: print(f"Sync failed: {e}")

async def ensure_session(x_session_id: str):
    if not x_session_id: return
    if x_session_id not in sessions:
        if db:
            doc = db.collection("python_sessions").document(x_session_id).get()
            if doc.exists:
                data = doc.to_dict()
                if "scaler_b64" in data:
                    try:
                        data["scaler"] = pickle.loads(base64.b64decode(data["scaler_b64"]))
                        del data["scaler_b64"]
                    except: pass
                if "df_records" in data:
                    data["df"] = pd.DataFrame(data["df_records"])
                    del data["df_records"]
                sessions[x_session_id] = data

def add_to_checklist(x_session_id: str, step_name: str):
    if x_session_id in sessions:
        if "audit" not in sessions[x_session_id]: sessions[x_session_id]["audit"] = {"execution_checklist": []}
        checklist = sessions[x_session_id]["audit"].get("execution_checklist", [])
        if step_name not in checklist: checklist.append(step_name)
        sessions[x_session_id]["audit"]["execution_checklist"] = checklist
        sync_session_to_firebase(x_session_id)

def get_representative_data(df):
    if len(df) <= 5: return df.to_dict(orient="records")
    return pd.concat([df.head(3), df.tail(2)]).to_dict(orient="records")

def get_weighted_x(X, weights_dict, features):
    if not weights_dict: return X
    w = np.array([weights_dict.get(f, 1.0) for f in features])
    return X * np.sqrt(w)

def calculate_ahp_weights_and_cr(matrix):
    n = len(matrix)
    col_sum = np.sum(matrix, axis=0)
    norm_matrix = matrix / (col_sum + 1e-10)
    weights = np.mean(norm_matrix, axis=1)
    aw = matrix @ weights
    λ_max = np.mean(aw / (weights + 1e-10))
    ci = (λ_max - n) / (n - 1) if n > 1 else 0
    ri_table = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    ri = ri_table.get(n, 1.49)
    return weights, (ci / ri if ri > 0 else 0)

def calculate_cluster_metrics(df, features, assignments, k, weights_dict=None):
    try:
        X_raw = df[features].select_dtypes(include=[np.number]).fillna(0).values
        X = get_weighted_x(X_raw, weights_dict, features)
        unique_labels = np.unique(assignments)
        dbi = float(davies_bouldin_score(X, assignments)) if len(unique_labels) > 1 else 0.0
        sil = float(silhouette_score(X, assignments)) if len(unique_labels) > 1 else 0.0
        chi = float(calinski_harabasz_score(X, assignments)) if len(unique_labels) > 1 else 0.0

        silhouette_values = []
        if len(unique_labels) > 1:
            sample_sil_values = silhouette_samples(X, assignments)
            for i in range(k):
                ith_cluster_sil_values = sample_sil_values[assignments == i]
                ith_cluster_sil_values.sort()
                silhouette_values.append({"cluster": int(i), "values": [float(v) for v in ith_cluster_sil_values], "avg": float(np.mean(ith_cluster_sil_values)) if len(ith_cluster_sil_values) > 0 else 0.0})

        profiles = {str(i): df[assignments == i][features].mean(numeric_only=True).to_dict() for i in range(k)}
        return {"davies_bouldin_index": dbi, "silhouette_score": sil, "calinski_harabasz_index": chi, "distribution": {str(i): {"count": int(np.sum(assignments == i)), "percentage": float(np.sum(assignments == i) / len(df) * 100)} for i in range(k)}, "cluster_profiles": profiles, "silhouette_plot_data": silhouette_values, "timestamp": time.time()}
    except Exception as e: return {"silhouette_score": 0.0}

def calculate_xie_beni(X, U, centers, m):
    n_samples = X.shape[0]
    dists_sq = np.sum((X[:, np.newaxis] - centers)**2, axis=2)
    numerator = np.sum((U**m).T * dists_sq)
    centers_dist_sq = np.sum((centers[:, np.newaxis] - centers)**2, axis=2)
    np.fill_diagonal(centers_dist_sq, np.inf)
    return float(numerator / (n_samples * np.min(centers_dist_sq) + 1e-10))

def calculate_partition_entropy(U):
    U_safe = np.fmax(U, 1e-10)
    return float(-np.sum(U * np.log(U_safe)) / U.shape[1])

@app.post("/stepwise/upload/")
async def stepwise_upload(file: UploadFile = File(...), x_session_id: Optional[str] = Header(None)):
    if not x_session_id: x_session_id = str(uuid.uuid4())
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(content))
        sessions[x_session_id] = {
            "df": df, "filename": file.filename, "config": {"filename": file.filename},
            "metrics": {}, "all_results": {}, "checkpoints": {"Data Asli": get_representative_data(df)},
            "audit": {"initial_rows": len(df), "initial_cols": len(df.columns), "execution_checklist": []}
        }
        sync_session_to_firebase(x_session_id)
        return {"status": "success", "jumlah_data": len(df), "columns": list(df.columns), "session_id": x_session_id}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/stepwise/raw-data/")
async def get_raw_data(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session expired")
    df = sessions[x_session_id]["df"]
    return {"columns": list(df.columns), "total_rows": int(len(df)), "data": pd.DataFrame(get_representative_data(df)).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records")}

@app.post("/stepwise/cleaning/")
async def stepwise_cleaning(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session expired")
    df = sessions[x_session_id]["df"].dropna(how='all').dropna(axis=1, how='all').drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns: df[col] = df[col].astype(str).str.strip()
    sessions[x_session_id]["df"] = df
    add_to_checklist(x_session_id, "Pembersihan Data")
    return {"status": "success", "final_rows": len(df)}

@app.post("/stepwise/missing-value/")
async def stepwise_missing(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session expired")
    df = sessions[x_session_id]["df"]
    for col in df.select_dtypes(include=['number']).columns: df[col] = df[col].fillna(df[col].median())
    sessions[x_session_id]["df"] = df
    add_to_checklist(x_session_id, "Imputasi Data")
    return {"status": "success"}

@app.get("/stepwise/normalization-stats/")
async def get_norm_stats(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session expired")
    df = sessions[x_session_id]["df"]
    num_df = df.select_dtypes(include=['number'])
    stats = {}
    for col in num_df.columns:
        series = num_df[col].dropna()
        if len(series) > 0:
            stats[col] = {
                "min": float(series.min()), "max": float(series.max()), "mean": float(series.mean()),
                "median": float(series.median()), "std": float(series.std()) if len(series) > 1 else 0.0,
                "variance": float(series.var()) if len(series) > 1 else 0.0
            }
    return {"status": "success", "stats": stats}

@app.post("/stepwise/robust-scaling/")
async def stepwise_robust_scaling(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session expired")
    from sklearn.preprocessing import RobustScaler
    df = sessions[x_session_id]["df"]
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        scaler = RobustScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        sessions[x_session_id]["df"] = df
        sessions[x_session_id]["scaler"] = scaler
        add_to_checklist(x_session_id, "Robust Scaling")
    return {"status": "success"}

@app.get("/stepwise/normality-test/")
async def stepwise_normality_test(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session expired")
    df = sessions[x_session_id]["df"]
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=[np.number]).columns))
    results = []
    non_normal = 0
    for f in features:
        data = df[f].fillna(0).values
        if len(data) > 3:
            stat, p = shapiro(data)
            results.append({"feature": f, "p_value": float(p), "is_normal": bool(p > 0.05)})
            if p <= 0.05: non_normal += 1
    return {"status": "success", "results": results, "recommendation": "RobustScaler" if non_normal > len(features)/2 else "StandardScaler", "justification": f"Ditemukan {non_normal} variabel tidak normal."}

@app.post("/stepwise/ahp-calculate/")
async def ahp_calculate(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body(...)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session expired")
    features = params.get("features")
    matrices_raw = params.get("matrices", [params.get("matrix")])
    matrices = [np.array(m) for m in matrices_raw if m]
    consensus_matrix = np.exp(np.mean(np.log(np.stack(matrices)), axis=0))
    weights, cr = calculate_ahp_weights_and_cr(consensus_matrix)
    weight_dict = {f: float(w) for f, w in zip(features, weights)}
    sessions[x_session_id]["config"]["ahp_weights"] = weight_dict
    sessions[x_session_id]["config"]["ahp_cr"] = float(cr)
    add_to_checklist(x_session_id, "AHP Konsensus")
    return {"status": "success", "weights": weight_dict, "consistency_ratio": float(cr), "is_consistent": cr < 0.1}

@app.post("/stepwise/fcm-init/")
async def fcm_init(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body(...)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session expired")
    k, m = params.get("k", 3), params.get("m", 2.0)
    df = sessions[x_session_id]["df"]
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=[np.number]).columns))
    X = get_weighted_x(df[features].fillna(0).values, sessions[x_session_id]["config"].get("ahp_weights"), features)
    U = np.random.dirichlet(np.ones(k), size=len(X)).T
    sessions[x_session_id]["algo_state"] = {"mode": "fcm", "U": U.tolist(), "X": X.tolist(), "features": features, "k": k, "m": m, "iteration": 0}
    add_to_checklist(x_session_id, "FCM Init")
    return {"status": "success", "message": "FCM diinisialisasi."}

@app.post("/stepwise/auto-converge/")
async def auto_converge(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    if not state: raise HTTPException(status_code=400, detail="Algo state missing")
    df = sessions[x_session_id]["df"]
    features, k = state["features"], state["k"]
    X = np.array(state["X"])
    if state.get("mode") == "fcm":
        U, m = np.array(state["U"]), state["m"]
        for _ in range(100):
            U_m = U ** m
            centers = (U_m @ X) / (U_m.sum(axis=1)[:, np.newaxis] + 1e-10)
            dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            dists = np.fmax(dists, 1e-10)
            inv_dists = dists ** (-2.0 / (m - 1))
            new_U = (inv_dists / inv_dists.sum(axis=1)[:, np.newaxis]).T
            if np.linalg.norm(new_U - U) < 1e-4: break
            U = new_U
        assignments = np.argmax(U, axis=0)
        metrics = calculate_cluster_metrics(df, features, assignments, k, sessions[x_session_id]["config"].get("ahp_weights"))
        metrics["centroids"] = centers.tolist()
        sessions[x_session_id]["metrics"] = metrics
        sessions[x_session_id]["df"]["cluster"] = assignments.tolist()
        return {"status": "success", "is_converged": True, "evaluation": metrics}
    return {"status": "error", "message": "K-Means auto-converge not implemented in local yet."}

@app.get("/stepwise/final-analysis/")
async def get_final_analysis(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session expired")
    session = sessions[x_session_id]
    metrics = session.get("metrics", {})
    return {"status": "success", "jumlah_data": len(session["df"]), "config": session.get("config", {}), "metrics": metrics, "silhouette_score": metrics.get("silhouette_score", 0.0), "davies_bouldin_index": metrics.get("davies_bouldin_index", 0.0), "hasil_cluster": session["df"].to_dict(orient="records")}

@app.post("/stepwise/ai-discussion/")
async def ai_discussion(x_session_id: Optional[str] = Header(None), lang: str = "id"):
    await ensure_session(x_session_id)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key: return {"status": "partial", "narrative": "AI not configured."}
    metrics = sessions[x_session_id].get("metrics", {})
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(f"Analyze student clusters: {str(metrics.get('cluster_profiles'))}. Language: {lang}.")
    return {"status": "success", "narrative": response.text}

@app.get("/stepwise/history-list/")
async def get_history():
    if not db: return {"status": "error", "message": "No DB"}
    docs = db.collection("python_sessions").stream()
    history = [{"session_id": doc.id, "filename": doc.to_dict().get("filename")} for doc in docs]
    return {"status": "success", "history": history}

@app.post("/stepwise/simulate-policy/")
async def simulate_policy(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body(...)):
    await ensure_session(x_session_id)
    session = sessions[x_session_id]
    df, metrics = session["df"], session["metrics"]
    target_cluster = params.get("target_cluster_idx")
    interventions = params.get("interventions", {})
    target_df = df[df["cluster"] == target_cluster].copy()
    features = metrics.get("feature_names", list(df.select_dtypes(include=[np.number]).columns))
    for f, pct in interventions.items():
        if f in target_df.columns: target_df[f] *= (1.0 + pct)
    X_new_raw = target_df[features].fillna(0).values
    X_new_scaled = session.get("scaler").transform(X_new_raw) if session.get("scaler") else X_new_raw
    X_new = get_weighted_x(X_new_scaled, session.get("config").get("ahp_weights"), features)
    centroids = np.array(metrics.get("centroids", []))
    if len(centroids) == 0: return {"status": "error", "message": "No centroids"}
    new_assignments = [int(np.argmin(np.linalg.norm(centroids - row, axis=1))) for row in X_new]
    migrated = sum(1 for a in new_assignments if a != target_cluster)
    return {"status": "success", "migrated_count": migrated, "total_impacted": len(target_df), "migration_rate": (migrated/len(target_df)*100) if len(target_df)>0 else 0}

@app.get("/stepwise/build-manuscript/")
async def build_manuscript(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    session = sessions[x_session_id]
    doc = Document()
    doc.add_heading('RESEARCH MANUSCRIPT: BORDERLAND STUDENTS', 0)
    doc.add_paragraph(f"Dataset: {session.get('filename')}")
    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return StreamingResponse(file_stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=Manuscript.docx"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
