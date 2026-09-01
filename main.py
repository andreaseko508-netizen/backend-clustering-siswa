from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Body, Query
from fastapi.responses import StreamingResponse
import os
import sys
import pandas as pd
import numpy as np
import time
import io
import uuid
import logging
import base64
from typing import Optional, List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# --- SYSTEM ARCHITECT LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMORBATAS-CORE")

# --- DYNAMIC PATH RECOVERY ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- PROFESSIONAL MODULE IMPORT ---
try:
    from utils import (
        sessions, audit_checkpoints, db, ensure_session,
        sync_session_to_firebase, add_to_checklist, get_representative_data
    )
    from statistics import (
        calculate_cluster_metrics, run_real_ga_init, get_weighted_x,
        perform_normality_test_expert, calculate_hopkins, calculate_ahp_weights_and_cr, safe_float
    )
    from reports import ResearchReportPDF
except ImportError:
    from api.utils import sessions, audit_checkpoints, db, ensure_session, sync_session_to_firebase, add_to_checklist, get_representative_data
    from api.statistics import calculate_cluster_metrics, run_real_ga_init, get_weighted_x, perform_normality_test_expert, calculate_hopkins, calculate_ahp_weights_and_cr, safe_float
    from api.reports import ResearchReportPDF

app = FastAPI(title="SIMORBATAS Enterprise AI Runtime", version="21.0.0")

# --- CORE ARCHITECT HELPERS ---
async def get_valid_session(x_session_id: str):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesi riset terputus. Silakan unggah dataset kembali.")
    return sessions[x_session_id]

def reorder_clusters_by_quality(df, features, assignments, k):
    """S2 Standard: Ensures C1 is always the 'Higher Performance' group."""
    try:
        scores = []
        for i in range(k):
            cluster_data = df[assignments == i][features]
            if not cluster_data.empty:
                score = cluster_data.mean().sum()
                scores.append((i, score))
            else:
                scores.append((i, -1e10))
        scores.sort(key=lambda x: x[1], reverse=True)
        remap = {old_id: new_id for new_id, (old_id, _) in enumerate(scores)}
        return np.array([remap[x] for x in assignments]), remap
    except Exception as e:
        logger.error(f"Ranking Sync Failed: {e}")
        return assignments, {i: i for i in range(k)}

@app.get("/")
async def root():
    return {"status": "Active", "engine": "SIMORBATAS-Vercel", "version": "21.0.0"}

# --- 1. DATA ACQUISITION & INTEGRITY ---

@app.post("/stepwise/upload/")
async def stepwise_upload(file: UploadFile = File(...), x_session_id: Optional[str] = Header(None)):
    if not x_session_id: x_session_id = str(uuid.uuid4())
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(content))
        df.columns = [str(c).strip() for c in df.columns]
        df = df.replace([np.inf, -np.inf], np.nan)
        sessions[x_session_id] = {
            "df": df, "filename": file.filename, "config": {"k": 3, "features": []},
            "start_time": time.time(), "metrics": {}, "checkpoints": {},
            "audit": {"execution_checklist": []}, "algo_state": {"iteration": 0, "history": []}
        }
        ensure_audit(x_session_id); audit_checkpoints[x_session_id]["01_Data_Asli"] = df.copy()
        sync_session_to_firebase(x_session_id)
        return {"status": "success", "session_id": x_session_id, "jumlah_data": len(df), "columns": list(df.columns)}
    except Exception as e:
        logger.error(f"Critical Upload Error: {e}")
        raise HTTPException(500, str(e))

@app.get("/stepwise/raw-data/")
async def get_raw_data(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    return {"columns": list(session["df"].columns), "total_rows": len(session["df"]), "data": pd.DataFrame(get_representative_data(session["df"])).fillna(0).to_dict(orient="records")}

# --- 2. PROFESSIONAL PREPROCESSING ---

@app.post("/stepwise/conversion/")
async def stepwise_conversion(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); df = session["df"]
    # Internal Ordinal Mapping Strategy
    ORD_RULES = {
        "prestasi": {"tidak pernah":0,"tingkat sekolah":1,"tingkat kecamatan":2,"tingkat kabupaten":3,"tingkat provinsi":4,"tingkat nasional":5,"tingkat internasional":6},
        "kendaraan": {"jalan kaki":0,"sepeda":1,"motor":2,"mobil":3,"angkutan umum":4}
    }
    mapping_report = {}
    for col in df.select_dtypes(include=['object']).columns:
        rule = next((v for k, v in ORD_RULES.items() if k in col.lower()), None)
        if rule:
            df[col] = df[col].astype(str).str.lower().str.strip().map(rule).fillna(0).astype(int)
        else:
            codes, uniques = pd.factorize(df[col])
            df[col] = codes
    session["df"] = df; add_to_checklist(x_session_id, "Konversi Fitur")
    return {"status": "success"}

@app.post("/stepwise/cleaning/")
async def stepwise_cleaning(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); df = session["df"].dropna(how='all').drop_duplicates()
    session["df"] = df; add_to_checklist(x_session_id, "Pembersihan Data")
    return {"status": "success", "final_rows": len(df)}

@app.post("/stepwise/missing-value/")
async def stepwise_missing(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    for col in session["df"].select_dtypes(include=['number']).columns:
        session["df"][col] = session["df"][col].fillna(session["df"][col].median())
    add_to_checklist(x_session_id, "Imputasi Data")
    return {"status": "success"}

# --- 3. ANALYTICAL AUDIT ---

@app.get("/stepwise/quality-report/")
async def get_quality_report(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); df = session["df"]
    num_cols = list(df.select_dtypes(include=['number']).columns)
    hopkins = calculate_hopkins(df[num_cols].fillna(0).values) if len(num_cols) >= 2 else 0.5
    return {"status": "success", "rows": len(df), "cols": len(df.columns), "hopkins_statistic": safe_float(hopkins), "is_suitable": len(df) > 5, "execution_checklist": session["audit"].get("execution_checklist", [])}

@app.get("/stepwise/normality-test/")
async def stepwise_normality_test(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); feats = session["config"].get("features", list(session["df"].columns))
    return perform_normality_test_expert(session["df"], feats)

@app.get("/stepwise/normalization-stats/")
async def get_norm_stats(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); num_df = session["df"].select_dtypes(include=['number'])
    stats = {col: {"min": safe_float(num_df[col].min()), "max": safe_float(num_df[col].max()), "mean": safe_float(num_df[col].mean()), "median": safe_float(num_df[col].median()), "std": safe_float(num_df[col].std()), "variance": safe_float(num_df[col].var())} for col in num_df.columns if not num_df[col].isnull().all()}
    return {"status": "success", "stats": stats}

@app.post("/stepwise/normalization/")
async def stepwise_norm(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); from sklearn.preprocessing import MinMaxScaler
    num_cols = session["df"].select_dtypes(include=['number']).columns
    if len(num_cols) > 0: session["df"][num_cols] = MinMaxScaler().fit_transform(session["df"][num_cols])
    add_to_checklist(x_session_id, "Normalisasi Data")
    return {"status": "success"}

# --- 4. ADVANCED K-MEANS CORE (GA OPTIMIZED) ---

@app.post("/stepwise/init-centroids/")
async def init_centroids_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    """PHASE 1: Genetic Algorithm Seed Discovery."""
    session = await get_valid_session(x_session_id); feats = session["config"]["features"]; k = 3
    X = get_weighted_x(session["df"][feats].fillna(0).values, session["config"].get("ahp_weights"), feats)
    # Perform Real GA Discovery
    best_seeds = run_real_ga_init(X, k, population_size=40, generations=50)
    session["algo_state"] = {"iteration": 0, "centroids": best_seeds.tolist(), "features": feats, "k": k, "history": []}
    add_to_checklist(x_session_id, "Inisialisasi GA"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "centroids": best_seeds.tolist(), "features": feats}

@app.post("/stepwise/calculate-distances/")
async def calculate_distances_step(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); state = session["algo_state"]; X = get_weighted_x(session["df"][state["features"]].fillna(0).values, session["config"].get("ahp_weights"), state["features"])
    dists = [np.linalg.norm(np.array(state["centroids"]) - row, axis=1).tolist() for row in X]
    state["distances"] = dists; return {"status": "success", "sample_work": {"distances": dists[0]}}

@app.post("/stepwise/assign-clusters/")
async def assign_clusters_step(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); state = session["algo_state"]; dists = np.array(state["distances"])
    assignments = np.argmin(dists, axis=1).tolist(); state["assignments"] = assignments
    add_to_checklist(x_session_id, "Cluster Assignment")
    return {"status": "success", "assignments": assignments}

@app.post("/stepwise/auto-converge/")
async def auto_converge(x_session_id: Optional[str] = Header(None)):
    """PHASE 2: Enterprise Grade Iteration & Rank Sync."""
    session = await get_valid_session(x_session_id); state = session["algo_state"]; ahp, k, feats = session["config"].get("ahp_weights"), state["k"], state["features"]
    X = get_weighted_x(session["df"][feats].fillna(0).values, ahp, feats)
    history = []; centroids = np.array(state["centroids"])
    for i in range(1, 101):
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2); assignments = np.argmin(dists, axis=1)
        wcss = safe_float(np.sum(np.min(dists, axis=1)**2)); new_c = np.array([X[assignments == j].mean(axis=0) if len(X[assignments == j]) > 0 else centroids[j] for j in range(k)])
        move = safe_float(np.linalg.norm(new_c - centroids)); history.append({"iter": i, "movement": move, "wcss": wcss}); centroids = new_c
        if move < 1e-4: break

    # S2 Rank Sync
    new_labels, remap = reorder_clusters_by_quality(session["df"], feats, assignments, k)
    final_centroids = [centroids[old_id].tolist() for old_id, _ in sorted(remap.items(), key=lambda x: x[1])]
    session["df"]["cluster"] = new_labels.tolist()
    for j in range(k): session["df"][f"dist_c{j}"] = np.linalg.norm(X - np.array(final_centroids[j]), axis=1)

    eval_k = calculate_cluster_metrics(session["df"], feats, new_labels, k, ahp)
    session.update({"metrics": eval_k, "algo_state": {**state, "centroids": final_centroids, "assignments": new_labels.tolist(), "is_converged": True, "history": history, "iteration": len(history)}})
    add_to_checklist(x_session_id, "Riset Selesai"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "is_converged": True, "iterations": len(history), "evaluation": eval_k, "hasil_cluster": session["df"].fillna(0).to_dict(orient="records")}

# --- 5. DATA EXPORTS ---

@app.get("/stepwise/final-analysis/")
async def get_final_analysis(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); m = session.get("metrics", {})
    return {"status": "success", "jumlah_data": len(session["df"]), "centroids": session.get("algo_state", {}).get("centroids", []), "config": session.get("config", {}), "metrics": m, "hasil_cluster": session["df"].fillna(0).to_dict(orient="records")}

@app.get("/stepwise/export-excel/")
async def export_excel_route(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_exp = session["df"].copy(); df_exp["cluster"] = df_exp["cluster"] + 1; df_exp.to_excel(writer, sheet_name="Hasil_Final", index=False)
    output.seek(0); return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
