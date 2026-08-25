from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Body
from fastapi.responses import StreamingResponse
import os
import sys
import pandas as pd
import numpy as np
import time
import io
import uuid
import logging
from typing import Optional, List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Configure logging for S2 Audit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMORBATAS")

# --- LOCAL MODULES ---
try:
    from .utils import (
        sessions, audit_checkpoints, db, ensure_session,
        sync_session_to_firebase, add_to_checklist, get_representative_data
    )
    from .statistics import (
        calculate_cluster_metrics, calculate_xie_beni, calculate_partition_entropy,
        calculate_hopkins, calculate_ahp_weights_and_cr, get_weighted_x, perform_significance_test,
        perform_stability_audit, perform_sensitivity_audit, perform_normality_test
    )
    from .reports import (
        ResearchReportPDF, generate_radar_chart_bytes, generate_bar_chart_bytes,
        generate_silhouette_chart_bytes, generate_manuscript_docx
    )
except ImportError:
    from utils import (
        sessions, audit_checkpoints, db, ensure_session,
        sync_session_to_firebase, add_to_checklist, get_representative_data
    )
    from statistics import (
        calculate_cluster_metrics, calculate_xie_beni, calculate_partition_entropy,
        calculate_hopkins, calculate_ahp_weights_and_cr, get_weighted_x, perform_significance_test,
        perform_stability_audit, perform_sensitivity_audit, perform_normality_test
    )
    from reports import (
        ResearchReportPDF, generate_radar_chart_bytes, generate_bar_chart_bytes,
        generate_silhouette_chart_bytes, generate_manuscript_docx
    )

app = FastAPI(title="SIMORBATAS Python AI Runtime", version="2.9.0")

@app.get("/")
async def root():
    return {"status": "Online", "engine": "SIMORBATAS-Runtime", "firebase": "Connected" if db else "Offline"}

@app.get("/health")
async def health():
    return {"status": "UP"}

# --- DATASET MANAGEMENT ---

@app.post("/stepwise/upload/")
async def stepwise_upload(file: UploadFile = File(...), x_session_id: Optional[str] = Header(None)):
    if not x_session_id: x_session_id = str(uuid.uuid4())
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(content))

        # S2 HARDENING: Clean column names and data
        df.columns = [str(c).strip() for c in df.columns]
        df = df.replace([np.inf, -np.inf], np.nan)

        sessions[x_session_id] = {
            "df": df, "filename": file.filename, "config": {"filename": file.filename},
            "metrics": {}, "all_results": {}, "checkpoints": {"Data Asli": get_representative_data(df)},
            "audit": {"execution_checklist": []}
        }

        if x_session_id not in audit_checkpoints: audit_checkpoints[x_session_id] = {}
        audit_checkpoints[x_session_id]["01_Data_Asli"] = df.copy()

        sync_session_to_firebase(x_session_id)
        return {"status": "success", "jumlah_data": len(df), "columns": list(df.columns), "session_id": x_session_id}
    except Exception as e:
        logger.error(f"Upload fail: {e}")
        raise HTTPException(500, str(e))

@app.get("/stepwise/raw-data/")
async def get_raw_data(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    df = sessions[x_session_id]["df"]
    return {"columns": list(df.columns), "total_rows": len(df), "data": pd.DataFrame(get_representative_data(df)).fillna(0).to_dict(orient="records")}

@app.post("/stepwise/mapping-config/")
async def stepwise_mapping(x_session_id: Optional[str] = Header(None), config: Dict[str, Any] = Body(...)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    sessions[x_session_id]["config"].update(config)
    if "features" in config:
        audit_checkpoints[x_session_id]["02_Seleksi_Variabel"] = sessions[x_session_id]["df"][config["features"]].copy()
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/stepwise/save_config/")
async def stepwise_save_config(x_session_id: Optional[str] = Header(None), config: Dict[str, Any] = Body(...)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    sessions[x_session_id]["config"].update(config)
    audit_checkpoints[x_session_id]["09_Konfigurasi_Algoritma"] = pd.DataFrame(list(config.items()))
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

# --- PREPROCESSING ---

@app.post("/stepwise/conversion/")
async def stepwise_conversion(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    df, config = sessions[x_session_id]["df"], sessions[x_session_id]["config"]
    mapping = {}
    cols = df.columns if not config.get("features") else config["features"]
    cat_cols = df[cols].select_dtypes(include=['object']).columns
    for col in cat_cols:
        codes, uniques = pd.factorize(df[col])
        df[col] = codes
        mapping[col] = {str(i): str(val) for i, val in enumerate(uniques)}
    sessions[x_session_id]["df"] = df
    audit_checkpoints[x_session_id]["03_Konversi_Kategori"] = df.copy()
    add_to_checklist(x_session_id, "Konversi Fitur")
    return {"status": "success", "mappings": mapping}

@app.post("/stepwise/cleaning/")
async def stepwise_cleaning(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    df = sessions[x_session_id]["df"].dropna(how='all').dropna(axis=1, how='all').drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns: df[col] = df[col].astype(str).str.strip()
    sessions[x_session_id]["df"] = df
    audit_checkpoints[x_session_id]["04_Data_Cleaning"] = df.copy()
    add_to_checklist(x_session_id, "Pembersihan Data")
    return {"status": "success", "final_rows": len(df)}

@app.post("/stepwise/missing-value/")
async def stepwise_missing(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    df = sessions[x_session_id]["df"]
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols: df[col] = df[col].fillna(df[col].median())
    sessions[x_session_id]["df"] = df
    audit_checkpoints[x_session_id]["05_Imputasi_Data"] = df.copy()
    add_to_checklist(x_session_id, "Imputasi Data")
    return {"status": "success"}

@app.get("/stepwise/normalization-stats/")
async def get_norm_stats(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    df = sessions[x_session_id]["df"]
    num_df = df.select_dtypes(include=['number'])
    stats = {col: {"min": float(num_df[col].min()), "max": float(num_df[col].max()), "mean": float(num_df[col].mean()), "median": float(num_df[col].median()), "std": float(num_df[col].std()) if len(num_df) > 1 else 0.0, "variance": float(num_df[col].var()) if len(num_df) > 1 else 0.0} for col in num_df.columns if not num_df[col].isnull().all()}
    return {"status": "success", "stats": stats}

@app.get("/stepwise/quality-report/")
async def get_quality_report(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    df = sessions[x_session_id]["df"]
    num_cols = list(df.select_dtypes(include=['number']).columns)
    total_cells = df.size
    null_count = df.isnull().sum().sum()
    completeness = 1.0 - (null_count / total_cells if total_cells > 0 else 0)
    return {
        "status": "success", "rows": len(df), "cols": len(df.columns),
        "numeric_features": len(num_cols), "completeness": float(completeness),
        "is_suitable": len(df) > 0 and len(num_cols) >= 2,
        "execution_checklist": sessions[x_session_id]["audit"].get("execution_checklist", [])
    }

@app.get("/stepwise/correlation-matrix/")
async def get_correlation_analysis(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    import base64
    import matplotlib.pyplot as plt
    import seaborn as sns
    df, config = sessions[x_session_id]["df"], sessions[x_session_id]["config"]
    features = config.get("features", list(df.select_dtypes(include=[np.number]).columns))
    num_df = df[features].select_dtypes(include=[np.number]).fillna(0)
    # S2 Fix: fillna(0) on corr matrix to prevent NaN in JSON
    corr_matrix = num_df.corr().fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    high_corr = [{"f1": corr_matrix.columns[i], "f2": corr_matrix.columns[j], "val": float(corr_matrix.iloc[i, j]), "interpretation": "Kuat"} for i in range(len(corr_matrix.columns)) for j in range(i) if abs(corr_matrix.iloc[i, j]) > 0.7]
    return {"status": "success", "heatmap_image": base64.b64encode(buf.getvalue()).decode('utf-8'), "high_correlations": high_corr}

# --- SCALING ---

@app.post("/stepwise/normalization/")
async def stepwise_norm(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    from sklearn.preprocessing import MinMaxScaler
    df = sessions[x_session_id]["df"]
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        sessions[x_session_id]["df"], sessions[x_session_id]["scaler"] = df, scaler
        audit_checkpoints[x_session_id]["07_Penskalaan_Fitur"] = df.copy()
        add_to_checklist(x_session_id, "Normalisasi Data")
    return {"status": "success"}

# --- ALGORITHMS (K-Means & FCM) ---

@app.post("/stepwise/init-centroids/")
async def init_centroids_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    df, k = sessions[x_session_id]["df"], params.get("k", 3)
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=[np.number]).columns))
    centroids = df[features].select_dtypes(include=[np.number]).fillna(0).sample(n=k, random_state=42).values.tolist()
    sessions[x_session_id]["algo_state"] = {"iteration": 0, "centroids": centroids, "features": features, "k": k, "history": []}
    audit_checkpoints[x_session_id]["10_Inisialisasi_Centroid"] = pd.DataFrame(centroids, columns=features)
    return {"status": "success", "centroids": centroids, "features": features}

@app.post("/stepwise/fcm-init/")
async def fcm_init_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3, "m": 2.0})):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    df, config = sessions[x_session_id]["df"], sessions[x_session_id]["config"]
    k, m = params.get("k", 3), params.get("m", 2.0)
    features = config.get("features", list(df.select_dtypes(include=[np.number]).columns))
    X = get_weighted_x(df[features].fillna(0).values, config.get("ahp_weights"), features)
    U = np.random.dirichlet(np.ones(k), size=len(X)).T
    sessions[x_session_id]["algo_state"] = {"mode": "fcm", "iteration": 0, "U": U.tolist(), "X": X.tolist(), "features": features, "k": k, "m": m, "history": []}
    return {"status": "success"}

@app.post("/stepwise/auto-converge/")
async def auto_converge(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    state = sessions[x_session_id]["algo_state"]
    if state.get("mode") == "fcm":
        X, U, m, k = np.array(state["X"]), np.array(state["U"]), state["m"], state["k"]
        for _ in range(100):
            centers = ((U**m) @ X) / ((U**m).sum(axis=1)[:, np.newaxis] + 1e-10)
            dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
            U = ( (dists ** (-2.0 / (m - 1))) / (dists ** (-2.0 / (m - 1))).sum(axis=1)[:, np.newaxis] ).T
        eval = calculate_cluster_metrics(sessions[x_session_id]["df"], state["features"], np.argmax(U, axis=0), k, sessions[x_session_id]["config"].get("ahp_weights"))
        sessions[x_session_id].update({"metrics": eval, "all_results": {"fcm": eval}})
        sync_session_to_firebase(x_session_id)
        return {"status": "success", "is_converged": True}
    return {"status": "error"}

@app.post("/stepwise/benchmark/")
@app.post("/stepwise/compare-all/")
async def stepwise_benchmark(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    session = sessions[x_session_id]
    df, config = session["df"], session["config"]
    X = get_weighted_x(df[config["features"]].fillna(0).values, config.get("ahp_weights"), config["features"])
    k = config.get("k", 3)
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    km_metrics = calculate_cluster_metrics(df, config["features"], km.labels_, k, config.get("ahp_weights"))
    # FCM
    m, U = config.get("m", 2.0), np.random.dirichlet(np.ones(k), size=len(X)).T
    for _ in range(50):
        centers = ((U**m) @ X) / ((U**m).sum(axis=1)[:, np.newaxis] + 1e-10)
        dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
        U = ( (dists ** (-2.0 / (m - 1))) / (dists ** (-2.0 / (m - 1))).sum(axis=1)[:, np.newaxis] ).T
    fcm_metrics = calculate_cluster_metrics(df, config["features"], np.argmax(U, axis=0), k, config.get("ahp_weights"))
    sig = perform_significance_test(X, km.labels_, np.argmax(U, axis=0))
    return {"status": "success", "comparison_data": {"kmeans": km_metrics, "fcm": fcm_metrics}, "significance": sig, "best_by_silhouette": "kmeans" if km_metrics["silhouette_score"] > fcm_metrics["silhouette_score"] else "fcm"}

@app.get("/stepwise/final-analysis/")
async def get_final_analysis(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(404, "Session not found")
    session = sessions[x_session_id]
    df_cleaned = session["df"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return {"status": "success", "config": session.get("config", {}), "metrics": session.get("metrics", {}), "hasil_cluster": df_cleaned.to_dict(orient="records")}

# --- HOUSEKEEPING ---

@app.get("/stepwise/checkpoints/")
async def get_checkpoints(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    return {"status": "success", "checkpoints": sessions[x_session_id].get("checkpoints", {})}

@app.get("/stepwise/session-state/")
async def get_session_state(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    return {"state": "UPLOADED" if x_session_id in sessions else "IDLE"}

@app.get("/stepwise/export-excel/")
async def export_excel_route(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in audit_checkpoints.get(x_session_id, {}).items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
