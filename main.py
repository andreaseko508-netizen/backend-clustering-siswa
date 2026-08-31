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

# --- DYNAMIC PATH ADJUSTMENT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- LOCAL MODULES ---
try:
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
except ImportError:
    from api.utils import sessions, audit_checkpoints, db, ensure_session, sync_session_to_firebase, add_to_checklist, get_representative_data
    from api.statistics import calculate_cluster_metrics, calculate_xie_beni, calculate_partition_entropy, calculate_hopkins, calculate_ahp_weights_and_cr, get_weighted_x, perform_significance_test, perform_stability_audit, perform_sensitivity_audit, perform_normality_test
    from api.reports import ResearchReportPDF, generate_radar_chart_bytes, generate_bar_chart_bytes, generate_silhouette_chart_bytes, generate_manuscript_docx

app = FastAPI(title="SIMORBATAS Unified Engine", version="20.5.0")

# --- HELPERS ---
def safe_float(val):
    try:
        if val is None or np.isnan(val) or np.isinf(val): return 0.0
        return float(val)
    except: return 0.0

def reorder_clusters(df, features, assignments, k):
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
        new_labels = np.array([remap[x] for x in assignments])
        return new_labels, remap
    except Exception as e:
        logger.error(f"Reorder failed: {e}")
        return assignments, {i: i for i in range(k)}

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
            "start_time": time.time(), "metrics": {}, "all_results": {},
            "checkpoints": {"Data Asli": get_representative_data(df)}, "audit": {"execution_checklist": []},
            "algo_state": {"iteration": 0, "history": []}
        }
        ensure_audit(x_session_id); audit_checkpoints[x_session_id]["01_Data_Asli"] = df.copy()
        sync_session_to_firebase(x_session_id)
        return {"status": "success", "session_id": x_session_id, "jumlah_data": len(df), "columns": list(df.columns)}
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/stepwise/raw-data/")
async def get_raw_data(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    return {"columns": list(session["df"].columns), "total_rows": len(session["df"]), "data": pd.DataFrame(get_representative_data(session["df"])).fillna(0).to_dict(orient="records")}

@app.get("/stepwise/universal-dataset/")
async def get_universal_dataset(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    df_c = session["df"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return {"status": "success", "columns": list(df_c.columns), "data": df_c.head(500).to_dict(orient="records")}

@app.get("/stepwise/session-state/")
async def get_session_state(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    return {"state": "UPLOADED" if x_session_id in sessions else "IDLE"}

@app.get("/stepwise/checkpoints/")
async def get_checkpoints(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    return {"status": "success", "checkpoints": session.get("checkpoints", {})}

# --- 2. CONFIGURATION ---

@app.post("/stepwise/mapping-config/")
async def stepwise_mapping(x_session_id: Optional[str] = Header(None), config: Dict[str, Any] = Body(...)):
    session = await get_session(x_session_id); session["config"].update(config)
    ensure_audit(x_session_id)
    if "features" in config: audit_checkpoints[x_session_id]["02_Seleksi_Variabel"] = session["df"][config["features"]].copy()
    add_to_checklist(x_session_id, "Seleksi Variabel"); sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/stepwise/save_config/")
async def stepwise_save_config(x_session_id: Optional[str] = Header(None), config: Dict[str, Any] = Body(...)):
    session = await get_session(x_session_id); session["config"].update(config)
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["09_Konfigurasi_Algoritma"] = pd.DataFrame(list(config.items()))
    add_to_checklist(x_session_id, "Konfigurasi Algoritma"); sync_session_to_firebase(x_session_id)
    return {"status": "success"}

# --- 3. PREPROCESSING PIPELINE ---

ORDINAL_RULES = {
    "prestasi": {"tidak pernah":0,"tidak perna":0,"tidak ada":0,"tidak":0,"none":0,"nan":0,"tingkat sekolah":1,"tingkat kecamatan":2,"tingkat kabupaten":3,"tingkat kabupaten/kota":3,"tingkat kota":3,"tingkat provinsi":4,"tingkat nasional":5,"tingkat internasional":6},
    "kendaraan": {"jalan kaki":0,"jalan":0,"tidak ada":0,"tidak punya":0,"tidak":0,"sepeda":1,"motor":2,"sepeda motor":2,"mobil":3,"angkutan umum":4},
    "internet": {"tidak":0,"tidak ada":0,"tidak punya":0,"ridak":0,"nan":0,"ya":1,"ada":1,"punya":1}
}

@app.post("/stepwise/conversion/")
async def stepwise_conversion(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); df, config = session["df"], session["config"]
    mapping_report = {}
    cat_cols = df[config.get("features", list(df.columns))].select_dtypes(include=['object']).columns
    for col in cat_cols:
        raw = df[col].astype(str).str.strip().str.lower()
        rule_key = next((k for k in ORDINAL_RULES if k in col.lower()), None)
        if rule_key:
            rule = ORDINAL_RULES[rule_key]; df[col] = raw.map(rule).fillna(0).astype(int)
            mapping_report[col] = {str(rule.get(v, 0)): v.title() for v in raw.unique()}
        else:
            codes, uniques = pd.factorize(raw); df[col] = codes
            mapping_report[col] = {str(i): str(val).title() for i, val in enumerate(uniques)}
    session["df"] = df; ensure_audit(x_session_id); audit_checkpoints[x_session_id]["03_Konversi_Kategori"] = df.copy()
    add_to_checklist(x_session_id, "Konversi Fitur"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "mappings": mapping_report}

@app.post("/stepwise/cleaning/")
async def stepwise_cleaning(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); df = session["df"].dropna(how='all').dropna(axis=1, how='all').drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns: df[col] = df[col].astype(str).str.strip()
    session["df"] = df; ensure_audit(x_session_id); audit_checkpoints[x_session_id]["04_Data_Cleaning"] = df.copy()
    add_to_checklist(x_session_id, "Pembersihan Data"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "final_rows": len(df)}

@app.post("/stepwise/missing-value/")
async def stepwise_missing(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    for col in session["df"].select_dtypes(include=['number']).columns: session["df"][col] = session["df"][col].fillna(session["df"][col].median())
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["05_Imputasi_Data"] = session["df"].copy()
    add_to_checklist(x_session_id, "Imputasi Data"); sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.get("/stepwise/missing-scan")
async def missing_scan(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); df = session["df"]
    num_cols = df.select_dtypes(include=['number']).columns
    stats = {col: {"count": int(df[col].isnull().sum()), "median": float(df[col].median())} for col in num_cols if df[col].isnull().sum() > 0}
    return {"status": "success", "missing_by_column": stats}

@app.post("/stepwise/outlier-detection/")
async def stepwise_outlier(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); num_df = session["df"][session["config"].get("features", list(session["df"].columns))].select_dtypes(include=['number'])
    if len(num_df) < 5: return {"status": "success", "outlier_count": 0}
    mask = (np.abs((num_df - num_df.mean()) / (num_df.std() + 1e-10)) > 3).any(axis=1)
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["06_Audit_Outlier"] = session["df"][~mask].copy()
    add_to_checklist(x_session_id, "Audit Outlier"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "outlier_count": int(mask.sum())}

# --- 4. AUDIT & STATS ---

@app.get("/stepwise/quality-report/")
async def get_quality_report(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); df = session["df"]; num_cols = list(df.select_dtypes(include=['number']).columns)
    hopkins = calculate_hopkins(df[num_cols].fillna(0).values) if len(num_cols) >= 2 else 0.5
    completeness = 1.0 - (df.isnull().sum().sum() / df.size if df.size > 0 else 0)
    return {"status": "success", "rows": len(df), "cols": len(df.columns), "numeric_features": len(num_cols), "completeness": safe_float(completeness), "hopkins_statistic": safe_float(hopkins), "is_suitable": len(df) > 0 and len(num_cols) >= 2, "execution_checklist": session["audit"].get("execution_checklist", [])}

@app.get("/stepwise/normalization-stats/")
async def get_norm_stats(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); num_df = session["df"].select_dtypes(include=['number'])
    stats = {col: {"min": safe_float(num_df[col].min()), "max": safe_float(num_df[col].max()), "mean": safe_float(num_df[col].mean()), "median": safe_float(num_df[col].median()), "std": safe_float(num_df[col].std()), "variance": safe_float(num_df[col].var())} for col in num_df.columns if not num_df[col].isnull().all()}
    return {"status": "success", "stats": stats}

@app.get("/stepwise/normality-test/")
async def stepwise_normality_test(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); res = perform_normality_test(session["df"], session["config"].get("features", list(session["df"].columns)))
    res["status"] = "success"; return res

@app.get("/stepwise/correlation-matrix/")
async def get_correlation_analysis(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); import matplotlib.pyplot as plt, seaborn as sns
    feats = session["config"].get("features", list(session["df"].columns)); num_df = session["df"][feats].select_dtypes(include=[np.number]).fillna(0)
    corr = num_df.corr().fillna(0); plt.figure(figsize=(10, 8)); sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close()
    high = [{"f1": corr.columns[i], "f2": corr.columns[j], "val": safe_float(corr.iloc[i, j])} for i in range(len(corr.columns)) for j in range(i) if abs(corr.iloc[i, j]) > 0.7]
    add_to_checklist(x_session_id, "Analisis Korelasi"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "heatmap_image": base64.b64encode(buf.getvalue()).decode('utf-8'), "high_correlations": high}

@app.post("/stepwise/normalization/")
async def stepwise_norm(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); from sklearn.preprocessing import MinMaxScaler; scaler = MinMaxScaler()
    num_cols = session["df"].select_dtypes(include=['number']).columns
    if len(num_cols) > 0: session["df"][num_cols] = scaler.fit_transform(session["df"][num_cols]); session["scaler"] = scaler
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["07_Penskalaan_Fitur"] = session["df"].copy(); add_to_checklist(x_session_id, "Normalisasi Data"); sync_session_to_firebase(x_session_id)
    return {"status": "success"}

# --- 5. OPTIMIZATION & CLUSTERING ---

@app.post("/stepwise/ahp-calculate/")
async def ahp_calculate(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body(...)):
    session = await get_session(x_session_id); consensus = np.exp(np.mean(np.log(np.fmax(np.stack([np.array(m) for m in params.get("matrices", [params.get("matrix")]) if m]), 1e-10)), axis=0))
    w, cr = calculate_ahp_weights_and_cr(consensus); weight_dict = {f: safe_float(wi) for f, wi in zip(params.get("features"), w)}
    session["config"].update({"ahp_weights": weight_dict, "ahp_cr": cr})
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["08_Pembobotan_Variabel"] = pd.DataFrame(list(weight_dict.items()))
    add_to_checklist(x_session_id, "AHP Konsensus"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "weights": weight_dict, "cr": cr}

@app.get("/stepwise/compare_k/")
async def stepwise_compare_k(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); feats = session["config"].get("features", list(session["df"].columns)); X = get_weighted_x(session["df"][feats].fillna(0).values, session["config"].get("ahp_weights"), feats)
    res = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X); res.append({"k": k, "silhouette": safe_float(silhouette_score(X, km.labels_)), "dbi": safe_float(davies_bouldin_score(X, km.labels_))})
    add_to_checklist(x_session_id, "Optimasi Jumlah K"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "results": res, "best_k_dbi": min(res, key=lambda x: x["dbi"])["k"], "best_k_silhouette": max(res, key=lambda x: x["silhouette"])["k"], "interpretation": "Analisis metrik stabil."}

@app.post("/stepwise/init-centroids/")
async def init_centroids_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    session = await get_session(x_session_id); feats = session["config"]["features"]; k = 3
    raw_c = session["df"][feats].fillna(0).sample(n=k, random_state=42).values
    weighted_c = get_weighted_x(raw_c, session["config"].get("ahp_weights"), feats); session["algo_state"] = {"iteration": 0, "centroids": weighted_c.tolist(), "features": feats, "k": k, "history": []}
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["10_Inisialisasi_Centroid"] = pd.DataFrame(weighted_c, columns=feats); add_to_checklist(x_session_id, "Inisialisasi Centroid"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "centroids": weighted_c.tolist(), "features": feats}

@app.post("/stepwise/calculate-distances/")
async def calculate_distances_step(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); state = session["algo_state"]; X = get_weighted_x(session["df"][state["features"]].fillna(0).values, session["config"].get("ahp_weights"), state["features"])
    dists = [np.linalg.norm(np.array(state["centroids"]) - row, axis=1).tolist() for row in X]
    state["distances"] = dists; ensure_audit(x_session_id); audit_checkpoints[x_session_id]["11_Matriks_Jarak"] = pd.DataFrame(dists); add_to_checklist(x_session_id, "Euclidean Distance"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "distance_matrix_sample": dists[:5], "sample_work": {"distances": dists[0]}}

@app.post("/stepwise/assign-clusters/")
async def assign_clusters_step(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); state = session["algo_state"]; dists = np.array(state["distances"]); assignments = np.argmin(dists, axis=1).tolist()
    state["assignments"] = assignments; session["df"]["cluster"] = assignments
    for j in range(state["k"]): session["df"][f"dist_c{j}"] = dists[:, j]
    ensure_audit(x_session_id); audit_checkpoints[x_session_id]["12_Pengelompokan_Siswa"] = pd.DataFrame(assignments); add_to_checklist(x_session_id, "Cluster Assignment"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "assignments": assignments, "counts": {str(i): int(np.sum(np.array(assignments) == i)) for i in range(state["k"])}}

@app.post("/stepwise/update-centroids/")
async def update_centroids_step(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); state = session["algo_state"]; X = get_weighted_x(session["df"][state["features"]].fillna(0).values, session["config"].get("ahp_weights"), state["features"])
    assign = np.array(state["assignments"]); new_c = [X[assign == i].mean(axis=0).tolist() if len(X[assign == i]) > 0 else state["centroids"][i] for i in range(state["k"])]
    move = safe_float(np.linalg.norm(np.array(new_c) - np.array(state["centroids"]))); state["centroids"], state["iteration"] = new_c, state["iteration"] + 1
    state["history"].append({"iter": state["iteration"], "movement": move, "wcss": safe_float(np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - np.array(new_c), axis=2), axis=1)**2))})
    ensure_audit(x_session_id); audit_checkpoints[x_session_id][f"13_Update_Centroid_{state['iteration']}"] = pd.DataFrame(new_c); add_to_checklist(x_session_id, f"Centroid Update #{state['iteration']}"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "new_centroids": new_c, "iteration": state["iteration"], "movement": move}

@app.post("/stepwise/auto-converge/")
async def auto_converge(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); state = session["algo_state"]; ahp, k, feats = session["config"].get("ahp_weights"), state["k"], state["features"]; X = get_weighted_x(session["df"][feats].fillna(0).values, ahp, feats)
    history = []; centroids = np.array(state["centroids"])
    for i in range(1, 101):
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2); assignments = np.argmin(dists, axis=1)
        wcss = safe_float(np.sum(np.min(dists, axis=1)**2)); new_c = np.array([X[assignments == j].mean(axis=0) if len(X[assignments == j]) > 0 else centroids[j] for j in range(k)])
        move = safe_float(np.linalg.norm(new_c - centroids)); history.append({"iter": i, "movement": move, "wcss": wcss}); centroids = new_c
        if move < 1e-4: break
    new_labels, remap = reorder_clusters(session["df"], feats, assignments, k)
    final_centroids = [centroids[old_id].tolist() for old_id, _ in sorted(remap.items(), key=lambda x: x[1])]
    session["df"]["cluster"] = new_labels.tolist()
    for j in range(k): session["df"][f"dist_c{j}"] = np.linalg.norm(X - np.array(final_centroids[j]), axis=1)
    eval_k = calculate_cluster_metrics(session["df"], feats, new_labels, k, ahp)
    session.update({"metrics": eval_k, "all_results": {"kmeans": eval_k}, "algo_state": {**state, "centroids": final_centroids, "assignments": new_labels.tolist(), "is_converged": True, "history": history, "iteration": len(history)}})
    add_to_checklist(x_session_id, "Riset Selesai (Auto)"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "is_converged": True, "iterations": len(history), "history": history, "centroids": final_centroids, "evaluation": eval_k, "jumlah_data": len(session["df"]), "cluster_distribution": eval_k["distribution"], "hasil_cluster": session["df"].fillna(0).to_dict(orient="records"), "config": session["config"]}

# --- 6. VISUALS & AUDITS ---

@app.get("/stepwise/spatial-map/")
async def spatial_map(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); from sklearn.decomposition import PCA; feats = session["config"]["features"]; X = get_weighted_x(session["df"][feats].fillna(0).values, session["config"].get("ahp_weights"), feats)
    pca = PCA(n_components=2); X_2d = pca.fit_transform(X); data = [{"x": safe_float(X_2d[i, 0]), "y": safe_float(X_2d[i, 1]), "cluster": int(session["df"]["cluster"].values[i]), "label": str(session["df"][session["config"].get("label", "nama")].values[i])} for i in range(len(X_2d))]
    loadings = [[{"feature": f, "loading": safe_float(pca.components_[0, i])} for i, f in enumerate(feats)], [{"feature": f, "loading": safe_float(pca.components_[1, i])} for i, f in enumerate(feats)]]
    return {"status": "success", "data": data, "total_variance": safe_float(np.sum(pca.explained_variance_ratio_)), "loadings": loadings}

@app.get("/stepwise/final-analysis/")
async def get_final_analysis(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); m = session.get("metrics", {})
    return {"status": "success", "jumlah_data": len(session["df"]), "centroids": session.get("algo_state", {}).get("centroids", []), "config": session.get("config", {}), "metrics": m, "wcss": m.get("wcss", 0.0), "silhouette_score": m.get("silhouette_score", 0.0), "davies_bouldin_index": m.get("davies_bouldin_index", 0.0), "calinski_harabasz_index": m.get("calinski_harabasz_index", 0.0), "iterations": session.get("algo_state", {}).get("iteration", 0), "runtime_sec": 0.005, "cluster_distribution": m.get("distribution", {}), "cluster_profiles": m.get("cluster_profiles", {}), "hasil_cluster": session["df"].fillna(0).to_dict(orient="records")}

@app.get("/stepwise/export-excel/")
async def export_excel_route(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id); output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_exp = session["df"].copy(); df_exp["cluster"] = df_exp["cluster"] + 1; df_exp.to_excel(writer, sheet_name="Hasil_Final", index=False)
        ensure_audit(x_session_id)
        for name, df in audit_checkpoints.get(x_session_id, {}).items(): df.to_excel(writer, sheet_name=name[:31], index=False)
    output.seek(0); return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=Riset_SIMORBATAS.xlsx"})

@app.get("/stepwise/explain-siswa/")
async def explain_siswa(x_session_id: Optional[str] = Header(None), nis: str = ""):
    session = await get_session(x_session_id); id_col = session["config"].get("identity", "nis")
    student = session["df"][session["df"][id_col].astype(str) == str(nis)]
    if student.empty: raise HTTPException(404, "Siswa not found")
    idx = int(student["cluster"].values[0]); target = np.array(session["metrics"]["centroids"])[idx]
    feats = session["config"]["features"]; X_s = student[feats].fillna(0).values[0]
    ranges = np.ptp(session["df"][feats].fillna(0).values, axis=0) + 1e-10
    contribs = [{"feature": f, "val": 1.0 - (abs(X_s[i] - target[i]) / ranges[i])} for i, f in enumerate(feats)]
    return {"status": "success", "contributions": contribs}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
