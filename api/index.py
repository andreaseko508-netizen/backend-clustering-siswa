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
import pickle
import base64
from typing import Optional, List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMORBATAS")

# --- LOCAL MODULES ---
try:
    from api.utils import (
        sessions, audit_checkpoints, db, ensure_session,
        sync_session_to_firebase, add_to_checklist, get_representative_data
    )
    from api.statistics import (
        calculate_cluster_metrics, calculate_xie_beni, calculate_partition_entropy,
        calculate_hopkins, calculate_ahp_weights_and_cr, get_weighted_x, perform_significance_test,
        perform_stability_audit, perform_sensitivity_audit, perform_normality_test
    )
    from api.reports import (
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

app = FastAPI(title="SIMORBATAS Research Engine", version="13.0.0")

# --- HELPERS ---
def safe_float(val):
    try:
        if val is None or np.isnan(val) or np.isinf(val): return 0.0
        return float(val)
    except: return 0.0

async def get_session(x_session_id: str):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesi riset tidak ditemukan. Silakan unggah dataset kembali.")
    return sessions[x_session_id]

def ensure_audit(x_session_id: str):
    if x_session_id not in audit_checkpoints:
        audit_checkpoints[x_session_id] = {}

@app.get("/")
async def root():
    return {"status": "Online", "engine": "SIMORBATAS-Vercel", "firebase": "Connected" if db else "Offline"}

@app.get("/health")
async def health():
    return {"status": "UP"}

# --- 1. DATASET MANAGEMENT ---

@app.post("/stepwise/upload/")
async def stepwise_upload(file: UploadFile = File(...), x_session_id: Optional[str] = Header(None)):
    if not x_session_id: x_session_id = str(uuid.uuid4())
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(content))
        df.columns = [str(c).strip() for c in df.columns]
        df = df.replace([np.inf, -np.inf], np.nan)

        sessions[x_session_id] = {
            "df": df, "filename": file.filename, "config": {"filename": file.filename},
            "start_time": time.time(),
            "metrics": {}, "all_results": {}, "checkpoints": {"Data Asli": get_representative_data(df)},
            "audit": {"execution_checklist": []},
            "algo_state": {"iteration": 0, "history": []}
        }
        ensure_audit(x_session_id)
        audit_checkpoints[x_session_id]["01_Data_Asli"] = df.copy()
        sync_session_to_firebase(x_session_id)
        return {"status": "success", "session_id": x_session_id, "jumlah_data": len(df), "columns": list(df.columns)}
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/stepwise/raw-data/")
async def get_raw_data(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    return {"columns": list(session["df"].columns), "total_rows": len(session["df"]), "data": pd.DataFrame(get_representative_data(session["df"])).fillna(0).to_dict(orient="records")}

@app.post("/stepwise/mapping-config/")
async def stepwise_mapping(x_session_id: Optional[str] = Header(None), config: Dict[str, Any] = Body(...)):
    session = await get_session(x_session_id)
    session["config"].update(config)
    ensure_audit(x_session_id)
    if "features" in config:
        audit_checkpoints[x_session_id]["02_Seleksi_Variabel"] = session["df"][config["features"]].copy()
    add_to_checklist(x_session_id, "Seleksi Variabel")
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/stepwise/save_config/")
async def stepwise_save_config(x_session_id: Optional[str] = Header(None), config: Dict[str, Any] = Body(...)):
    session = await get_session(x_session_id)
    session["config"].update(config)
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id]["09_Konfigurasi_Algoritma"] = pd.DataFrame(list(config.items()))
    add_to_checklist(x_session_id, "Konfigurasi Algoritma")
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

# --- 2. PREPROCESSING ---

ORDINAL_RULES = {
    "prestasi": {"tidak pernah":0,"tidak perna":0,"tidak ada":0,"tidak":0,"none":0,"nan":0,"tingkat sekolah":1,"tingkat kecamatan":2,"tingkat kabupaten":3,"tingkat kabupaten/kota":3,"tingkat kota":3,"tingkat provinsi":4,"tingkat nasional":5,"tingkat internasional":6},
    "kendaraan": {"jalan kaki":0,"jalan":0,"tidak ada":0,"tidak punya":0,"tidak":0,"sepeda":1,"motor":2,"sepeda motor":2,"mobil":3,"angkutan umum":4},
    "internet": {"tidak":0,"tidak ada":0,"tidak punya":0,"ridak":0,"nan":0,"ya":1,"ada":1,"punya":1}
}

@app.post("/stepwise/conversion/")
async def stepwise_conversion(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    df, config = session["df"], session["config"]
    mapping_report = {}
    cat_cols = df[config.get("features", list(df.columns))].select_dtypes(include=['object']).columns
    for col in cat_cols:
        raw = df[col].astype(str).str.strip().str.lower()
        rule_key = next((k for k in ORDINAL_RULES if k in col.lower()), None)
        if rule_key:
            rule = ORDINAL_RULES[rule_key]
            df[col] = raw.map(rule).fillna(0).astype(int)
            mapping_report[col] = {str(rule.get(v, 0)): v.title() for v in raw.unique()}
        else:
            codes, uniques = pd.factorize(raw)
            df[col] = codes
            mapping_report[col] = {str(i): str(val).title() for i, val in enumerate(uniques)}
    session["df"] = df
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["03_Konversi_Kategori"] = df.copy()
    add_to_checklist(x_session_id, "Konversi Fitur")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "mappings": mapping_report}

@app.post("/stepwise/cleaning/")
async def stepwise_cleaning(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    df = session["df"].dropna(how='all').dropna(axis=1, how='all').drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns: df[col] = df[col].astype(str).str.strip()
    session["df"] = df
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["04_Data_Cleaning"] = df.copy()
    add_to_checklist(x_session_id, "Pembersihan Data")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "final_rows": len(df)}

@app.post("/stepwise/missing-value/")
async def stepwise_missing(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    for col in session["df"].select_dtypes(include=['number']).columns: session["df"][col] = session["df"][col].fillna(session["df"][col].median())
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["05_Imputasi_Data"] = session["df"].copy()
    add_to_checklist(x_session_id, "Imputasi Data")
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/stepwise/outlier-detection/")
async def stepwise_outlier(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    feats = session["config"].get("features", list(session["df"].columns))
    num_df = session["df"][feats].select_dtypes(include=['number'])
    if len(num_df) < 5: return {"status": "success", "outlier_count": 0}
    mask = (np.abs((num_df - num_df.mean()) / (num_df.std() + 1e-10)) > 3).any(axis=1)
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["06_Audit_Outlier"] = session["df"][~mask].copy()
    add_to_checklist(x_session_id, "Audit Outlier")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "outlier_count": int(mask.sum())}

# --- 3. STATS & AUDIT ---

@app.get("/stepwise/quality-report/")
async def get_quality_report(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    num_cols = list(session["df"].select_dtypes(include=['number']).columns)
    hopkins = calculate_hopkins(session["df"][num_cols].fillna(0).values) if len(num_cols) >= 2 else 0.5
    completeness = 1.0 - (session["df"].isnull().sum().sum() / session["df"].size if session["df"].size > 0 else 0)
    return {"status": "success", "rows": len(session["df"]), "cols": len(session["df"].columns), "numeric_features": len(num_cols), "completeness": safe_float(completeness), "hopkins_statistic": safe_float(hopkins), "is_suitable": len(session["df"]) > 0 and len(num_cols) >= 2, "execution_checklist": session["audit"].get("execution_checklist", [])}

@app.get("/stepwise/correlation-matrix/")
async def get_correlation_analysis(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    import matplotlib.pyplot as plt, seaborn as sns
    feats = session["config"].get("features", list(session["df"].columns))
    num_df = session["df"][feats].select_dtypes(include=[np.number]).fillna(0)
    corr = num_df.corr().fillna(0)
    plt.figure(figsize=(10, 8)); sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close()
    high = [{"f1": corr.columns[i], "f2": corr.columns[j], "val": safe_float(corr.iloc[i, j])} for i in range(len(corr.columns)) for j in range(i) if abs(corr.iloc[i, j]) > 0.7]
    add_to_checklist(x_session_id, "Analisis Korelasi")
    return {"status": "success", "heatmap_image": base64.b64encode(buf.getvalue()).decode('utf-8'), "high_correlations": high}

@app.post("/stepwise/normalization/")
async def stepwise_norm(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    num_cols = session["df"].select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        session["df"][num_cols] = scaler.fit_transform(session["df"][num_cols])
        session["scaler"] = scaler
        ensure_audit(x_session_id); audit_checkpoints[x_session_id]["07_Penskalaan_Fitur"] = session["df"].copy()
        add_to_checklist(x_session_id, "Normalisasi Data")
        sync_session_to_firebase(x_session_id)
    return {"status": "success"}

# --- 4. CLUSTERING ALGORITHMS ---

@app.post("/stepwise/ahp-calculate/")
async def ahp_calculate(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body(...)):
    session = await get_session(x_session_id)
    consensus = np.exp(np.mean(np.log(np.fmax(np.stack([np.array(m) for m in params.get("matrices", [params.get("matrix")]) if m]), 1e-10)), axis=0))
    w, cr = calculate_ahp_weights_and_cr(consensus)
    weight_dict = {f: safe_float(wi) for f, wi in zip(params.get("features"), w)}
    session["config"].update({"ahp_weights": weight_dict, "ahp_cr": cr})
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["08_Pembobotan_Variabel"] = pd.DataFrame(list(weight_dict.items()))
    add_to_checklist(x_session_id, "AHP Konsensus")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "weights": weight_dict, "cr": cr}

@app.get("/stepwise/compare_k/")
async def stepwise_compare_k(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    feats = session["config"].get("features", list(session["df"].columns))
    X = get_weighted_x(session["df"][feats].fillna(0).values, session["config"].get("ahp_weights"), feats)
    res = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        res.append({"k": k, "silhouette": safe_float(silhouette_score(X, km.labels_)), "dbi": safe_float(davies_bouldin_score(X, km.labels_))})
    add_to_checklist(x_session_id, "Optimasi Jumlah K")
    return {"status": "success", "results": res, "best_k_dbi": min(res, key=lambda x: x["dbi"])["k"]}

@app.post("/stepwise/init-centroids/")
async def init_centroids_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    session = await get_session(x_session_id)
    feats, k = session["config"]["features"], params.get("k", 3)
    raw_c = session["df"][feats].fillna(0).sample(n=k, random_state=42).values
    weighted_c = get_weighted_x(raw_c, session["config"].get("ahp_weights"), feats)
    session["algo_state"] = {"iteration": 0, "centroids": weighted_c.tolist(), "features": feats, "k": k, "history": []}
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["10_Inisialisasi_Centroid"] = pd.DataFrame(weighted_c, columns=feats)
    add_to_checklist(x_session_id, "Inisialisasi Centroid")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "centroids": weighted_c.tolist()}

@app.post("/stepwise/assign-clusters/")
async def assign_clusters_step(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    state = session["algo_state"]
    X = get_weighted_x(session["df"][state["features"]].fillna(0).values, session["config"].get("ahp_weights"), state["features"])
    dists = np.linalg.norm(X[:, np.newaxis] - np.array(state["centroids"]), axis=2)
    assignments = np.argmin(dists, axis=1).tolist()
    state["assignments"] = assignments
    session["df"]["cluster"] = assignments
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["12_Pengelompokan_Siswa"] = pd.DataFrame(assignments)
    add_to_checklist(x_session_id, "Cluster Assignment")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "assignments": assignments, "counts": {str(i): int(np.sum(np.array(assignments) == i)) for i in range(state["k"])}}

@app.post("/stepwise/auto-converge/")
async def auto_converge(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    state = session["algo_state"]
    ahp, k, feats = session["config"].get("ahp_weights"), state["k"], state["features"]
    X = get_weighted_x(session["df"][feats].fillna(0).values, ahp, feats)
    history = []

    if state.get("mode") == "fcm":
        U, m = np.array(state["U"]), state["m"]
        for i in range(1, 101):
            U_m = U ** m
            centers = (U_m @ X) / (U_m.sum(axis=1)[:, np.newaxis] + 1e-10)
            dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
            new_U = ((dists**(-2./(m-1))) / (dists**(-2./(m-1))).sum(axis=1)[:, np.newaxis]).T
            diff = safe_float(np.linalg.norm(new_U - U))
            history.append({"iter": i, "movement": diff, "wcss": safe_float(np.sum((U_m).T * (dists**2)))})
            U = new_U
            if diff < 1e-4: break
        labels = np.argmax(U, axis=0)
        session["df"]["cluster"] = labels.tolist()
        eval_f = calculate_cluster_metrics(session["df"], feats, labels, k, ahp)
        session.update({"metrics": eval_f, "all_results": {"fcm": eval_f}, "algo_state": {**state, "U": U.tolist(), "centroids": centers.tolist(), "is_converged": True, "history": history, "iteration": len(history)}})
    else:
        centroids = np.array(state["centroids"])
        for i in range(1, 101):
            dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            assignments = np.argmin(dists, axis=1)
            new_c = np.array([X[assignments == j].mean(axis=0) if len(X[assignments == j]) > 0 else centroids[j] for j in range(k)])
            move = safe_float(np.linalg.norm(new_c - centroids))
            history.append({"iter": i, "movement": move, "wcss": safe_float(np.sum(np.min(dists, axis=1)**2))})
            centroids = new_c
            if move < 1e-4: break
        session["df"]["cluster"] = assignments.tolist()
        eval_k = calculate_cluster_metrics(session["df"], feats, assignments, k, ahp)
        session.update({"metrics": eval_k, "all_results": {"kmeans": eval_k}, "algo_state": {**state, "centroids": centroids.tolist(), "assignments": assignments.tolist(), "is_converged": True, "history": history, "iteration": len(history)}})

    add_to_checklist(x_session_id, "Riset Selesai (Auto)")
    sync_session_to_firebase(x_session_id)
    # Return flattened
    resp = {"status": "success", "is_converged": True, "iterations": len(history), "history": history, "centroids": session["algo_state"]["centroids"]}
    resp.update(session["metrics"])
    resp["runtime_sec"] = time.time() - session.get("start_time", time.time())
    resp["jumlah_data"] = len(session["df"])
    resp["cluster_distribution"] = session["metrics"]["distribution"]
    resp["hasil_cluster"] = session["df"].fillna(0).to_dict(orient="records")
    return resp

# --- 5. ANALYTICS ---

@app.get("/stepwise/final-analysis/")
async def get_final_analysis(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    m = session.get("metrics", {})
    return {
        "status": "success", "jumlah_data": len(session["df"]),
        "wcss": m.get("wcss", 0.0), "silhouette_score": m.get("silhouette_score", 0.0),
        "davies_bouldin_index": m.get("davies_bouldin_index", 0.0),
        "calinski_harabasz_index": m.get("calinski_harabasz_index", 0.0),
        "iterations": session.get("algo_state", {}).get("iteration", 0),
        "runtime_sec": safe_float(time.time() - session.get("start_time", time.time())),
        "cluster_distribution": m.get("distribution", {}),
        "cluster_profiles": m.get("cluster_profiles", {}),
        "hasil_cluster": session["df"].fillna(0).to_dict(orient="records"),
        "config": session.get("config", {})
    }

@app.get("/stepwise/export-excel/")
async def export_excel_route(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        session["df"].to_excel(writer, sheet_name="Hasil_Final", index=False)
        ensure_audit(x_session_id)
        for name, df in audit_checkpoints.get(x_session_id, {}).items(): df.to_excel(writer, sheet_name=name[:31], index=False)
    output.seek(0); return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=Riset_SIMORBATAS.xlsx"})

@app.get("/stepwise/export-pdf/")
async def export_pdf_route(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    pdf = ResearchReportPDF(); pdf.add_page(); pdf.chapter_title("LAPORAN RISET")
    return StreamingResponse(io.BytesIO(pdf.output()), media_type="application/pdf")

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
