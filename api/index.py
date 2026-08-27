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

app = FastAPI(title="SIMORBATAS Final AI Engine", version="7.0.0")

@app.get("/")
async def root():
    return {"status": "Online", "engine": "SIMORBATAS-Vercel", "firebase": "Connected" if db else "Offline"}

@app.get("/health")
async def health():
    return {"status": "UP"}

# --- HELPERS ---
async def get_session(x_session_id: str):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        logger.error(f"Session {x_session_id} not found in RAM/Cloud")
        raise HTTPException(status_code=404, detail="Sesi riset tidak ditemukan. Silakan unggah dataset kembali.")
    return sessions[x_session_id]

def ensure_audit(x_session_id: str):
    """Ensures audit dictionary exists for this session."""
    if x_session_id not in audit_checkpoints:
        audit_checkpoints[x_session_id] = {}

# --- 1. DATASET & CONFIG ---

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
            "metrics": {}, "all_results": {}, "checkpoints": {"Data Asli": get_representative_data(df)},
            "audit": {"execution_checklist": []}
        }
        ensure_audit(x_session_id)
        audit_checkpoints[x_session_id]["01_Data_Asli"] = df.copy()

        sync_session_to_firebase(x_session_id)
        return {"status": "success", "session_id": x_session_id, "jumlah_data": len(df), "columns": list(df.columns)}
    except Exception as e:
        logger.error(f"Upload fail: {e}")
        raise HTTPException(500, str(e))

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

@app.get("/stepwise/session-state/")
async def get_session_state(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    return {"state": "UPLOADED" if x_session_id in sessions else "IDLE"}

# --- 2. PREPROCESSING ---

@app.post("/stepwise/conversion/")
async def stepwise_conversion(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    df, config = session["df"], session["config"]
    cat_cols = df[config.get("features", list(df.columns))].select_dtypes(include=['object']).columns
    mapping = {}
    for col in cat_cols:
        codes, uniques = pd.factorize(df[col])
        df[col] = codes
        mapping[col] = {str(i): str(val) for i, val in enumerate(uniques)}
    session["df"] = df
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id]["03_Konversi_Kategori"] = df.copy()
    add_to_checklist(x_session_id, "Konversi Fitur")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "mappings": mapping}

@app.post("/stepwise/cleaning/")
async def stepwise_cleaning(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    df = session["df"].dropna(how='all').dropna(axis=1, how='all').drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns: df[col] = df[col].astype(str).str.strip()
    session["df"] = df
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id]["04_Data_Cleaning"] = df.copy()
    add_to_checklist(x_session_id, "Pembersihan Data")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "final_rows": len(df)}

@app.post("/stepwise/missing-value/")
async def stepwise_missing(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    df = session["df"]
    for col in df.select_dtypes(include=['number']).columns: df[col] = df[col].fillna(df[col].median())
    session["df"] = df
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id]["05_Imputasi_Data"] = df.copy()
    add_to_checklist(x_session_id, "Imputasi Data")
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/stepwise/outlier-detection/")
async def stepwise_outlier(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    num_df = session["df"][session["config"].get("features", list(session["df"].columns))].select_dtypes(include=['number'])
    if len(num_df) < 5: return {"status": "success", "outlier_count": 0}
    mask = (np.abs((num_df - num_df.mean()) / (num_df.std() + 1e-10)) > 3).any(axis=1)
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id]["06_Audit_Outlier"] = session["df"][~mask].copy()
    add_to_checklist(x_session_id, "Audit Outlier")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "outlier_count": int(mask.sum())}

# --- 3. AUDIT & STATS ---

@app.get("/stepwise/quality-report/")
async def get_quality_report(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    df = session["df"]
    num_cols = list(df.select_dtypes(include=['number']).columns)
    hopkins = calculate_hopkins(df[num_cols].fillna(0).values) if len(num_cols) >= 2 else 0.5
    completeness = 1.0 - (df.isnull().sum().sum() / df.size if df.size > 0 else 0)
    return {"status": "success", "rows": len(df), "cols": len(df.columns), "numeric_features": len(num_cols), "completeness": float(completeness), "hopkins_statistic": hopkins, "is_suitable": len(df) > 0 and len(num_cols) >= 2, "execution_checklist": session["audit"].get("execution_checklist", [])}

@app.get("/stepwise/normalization-stats/")
async def get_norm_stats(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    num_df = session["df"].select_dtypes(include=['number'])
    stats = {col: {"min": float(num_df[col].min()), "max": float(num_df[col].max()), "mean": float(num_df[col].mean()), "median": float(num_df[col].median()), "std": float(num_df[col].std()) if len(num_df) > 1 else 0.0, "variance": float(num_df[col].var()) if len(num_df) > 1 else 0.0} for col in num_df.columns if not num_df[col].isnull().all()}
    return {"status": "success", "stats": stats}

@app.get("/stepwise/normality-test/")
async def stepwise_normality_test(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    res = perform_normality_test(session["df"], session["config"].get("features", list(session["df"].columns)))
    res["status"] = "success"
    return res

@app.get("/stepwise/correlation-matrix/")
async def get_correlation_analysis(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    import base64, matplotlib.pyplot as plt, seaborn as sns
    num_df = session["df"][session["config"].get("features", list(session["df"].columns))].select_dtypes(include=[np.number]).fillna(0)
    corr = num_df.corr().fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    high = [{"f1": corr.columns[i], "f2": corr.columns[j], "val": float(corr.iloc[i, j])} for i in range(len(corr.columns)) for j in range(i) if abs(corr.iloc[i, j]) > 0.7]
    add_to_checklist(x_session_id, "Analisis Korelasi")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "heatmap_image": base64.b64encode(buf.getvalue()).decode('utf-8'), "high_correlations": high}

# --- 4. SCALING ---

@app.post("/stepwise/normalization/")
async def stepwise_norm(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    num_cols = session["df"].select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        session["df"][num_cols] = scaler.fit_transform(session["df"][num_cols])
        session["scaler"] = scaler
        ensure_audit(x_session_id)
        audit_checkpoints[x_session_id]["07_Penskalaan_Fitur"] = session["df"].copy()
        add_to_checklist(x_session_id, "Normalisasi Data")
        sync_session_to_firebase(x_session_id)
    return {"status": "success"}

# --- 5. OPTIMIZATION & AHP ---

@app.post("/stepwise/ahp-calculate/")
async def ahp_calculate(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body(...)):
    session = await get_session(x_session_id)
    consensus = np.exp(np.mean(np.log(np.fmax(np.stack([np.array(m) for m in params.get("matrices", [params.get("matrix")]) if m]), 1e-10)), axis=0))
    w, cr = calculate_ahp_weights_and_cr(consensus)
    weight_dict = {f: float(np.nan_to_num(wi)) for f, wi in zip(params.get("features"), w)}
    session["config"].update({"ahp_weights": weight_dict, "ahp_cr": cr})
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id]["08_Pembobotan_Variabel"] = pd.DataFrame(list(weight_dict.items()))
    add_to_checklist(x_session_id, "AHP Konsensus")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "weights": weight_dict, "cr": cr}

@app.post("/stepwise/elbow/")
async def stepwise_elbow(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    X = session["df"].select_dtypes(include=[np.number]).fillna(0).values
    wcss = [{"k": i, "wcss": float(KMeans(n_clusters=i, n_init=10, random_state=42).fit(X).inertia_)} for i in range(1, 11)]
    add_to_checklist(x_session_id, "Analisis Elbow")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "data": wcss}

@app.get("/stepwise/compare_k/")
async def stepwise_compare_k(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    feats = session["config"].get("features", list(session["df"].columns))
    X = get_weighted_x(session["df"][feats].fillna(0).values, session["config"].get("ahp_weights"), feats)
    results = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X)
        results.append({
            "k": k,
            "silhouette": float(silhouette_score(X, km.labels_)),
            "dbi": float(davies_bouldin_score(X, km.labels_))
        })

    best_k_sil = max(results, key=lambda x: x["silhouette"])["k"]
    best_k_dbi = min(results, key=lambda x: x["dbi"])["k"]

    add_to_checklist(x_session_id, "Optimasi Jumlah K")
    sync_session_to_firebase(x_session_id)
    return {
        "status": "success",
        "results": results,
        "best_k_dbi": best_k_dbi,
        "best_k_silhouette": best_k_sil,
        "interpretation": f"Berdasarkan validasi metrik, K={best_k_sil} memiliki kepadatan terbaik (Silhouette), sedangkan K={best_k_dbi} memiliki pemisahan klaster terbaik (DBI)."
    }

@app.post("/stepwise/gap-statistic/")
async def stepwise_gap_statistic(x_session_id: Optional[str] = Header(None)):
    """Step 15: Gap Statistic for K selection justification."""
    session = await get_session(x_session_id)
    X = session["df"].select_dtypes(include=[np.number]).fillna(0).values
    if len(X) < 10: return {"status": "success", "recommended_k": 3, "gap_values": []}

    n_samples, n_features = X.shape
    ks = range(1, 7)
    gaps = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X)
        log_wcss = np.log(km.inertia_ + 1e-10)
        ref_log_wcss = [np.log(KMeans(n_clusters=k, n_init=5, random_state=i).fit(np.random.uniform(X.min(axis=0), X.max(axis=0), size=(n_samples, n_features))).inertia_ + 1e-10) for i in range(5)]
        gaps.append({"k": k, "gap": float(np.mean(ref_log_wcss) - log_wcss)})

    recommended_k = int(ks[np.argmax([g["gap"] for g in gaps])])
    add_to_checklist(x_session_id, "Gap Statistic")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "gap_values": gaps, "recommended_k": recommended_k}

# --- 6. K-MEANS LOGIC ---

@app.post("/stepwise/init-centroids/")
async def init_centroids_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    session = await get_session(x_session_id)
    df, k = session["df"], params.get("k", 3)
    feats = session["config"]["features"]
    centroids = df[feats].fillna(0).sample(n=k, random_state=42).values.tolist()
    session["algo_state"] = {"iteration": 0, "centroids": centroids, "features": feats, "k": k, "history": []}
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id]["10_Inisialisasi_Centroid"] = pd.DataFrame(centroids, columns=feats)
    add_to_checklist(x_session_id, "Inisialisasi Centroid")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "centroids": centroids, "features": feats}

@app.post("/stepwise/init-centroids-ga/")
async def init_ga_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    session = await get_session(x_session_id)
    feats, k = session["config"]["features"], params.get("k", 3)
    X = get_weighted_x(session["df"][feats].fillna(0).values, session["config"].get("ahp_weights"), feats)
    from sklearn.cluster import kmeans_plusplus
    c, _ = kmeans_plusplus(X, n_clusters=k, random_state=42)
    session["algo_state"] = {"iteration": 0, "centroids": c.tolist(), "features": feats, "k": k, "history": []}
    add_to_checklist(x_session_id, "Inisialisasi Centroid GA")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "centroids": c.tolist(), "features": feats}

@app.post("/stepwise/calculate-distances/")
async def calculate_distances_step(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    state, config = session["algo_state"], session["config"]
    X = get_weighted_x(session["df"][state["features"]].fillna(0).values, config.get("ahp_weights"), state["features"])
    dists = [np.linalg.norm(np.array(state["centroids"]) - row, axis=1).tolist() for row in X]
    state["distances"] = dists
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id]["11_Matriks_Jarak"] = pd.DataFrame(dists)
    add_to_checklist(x_session_id, "Euclidean Distance")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "distance_matrix_sample": dists[:5], "sample_work": {"distances": dists[0]}}

@app.post("/stepwise/assign-clusters/")
async def assign_clusters_step(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    state = session["algo_state"]
    dists = np.array(state["distances"])
    assignments = np.argmin(dists, axis=1).tolist()
    state["assignments"] = assignments
    state["current_wcss"] = float(np.sum(np.min(dists, axis=1)**2))
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id]["12_Pengelompokan_Siswa"] = pd.DataFrame(assignments)
    add_to_checklist(x_session_id, "Cluster Assignment")
    sync_session_to_firebase(x_session_id)
    counts = {str(i): int(np.sum(np.array(assignments) == i)) for i in range(state["k"])}
    return {"status": "success", "assignments": assignments, "counts": counts, "wcss": state["current_wcss"]}

@app.post("/stepwise/update-centroids/")
async def update_centroids_step(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    state = session["algo_state"]
    df, assignments = session["df"][state["features"]].fillna(0), np.array(state["assignments"])
    new_centers = [df[assignments == i].mean(axis=0).values.tolist() if len(df[assignments == i]) > 0 else state["centroids"][i] for i in range(state["k"])]
    movement = float(np.linalg.norm(np.array(new_centers) - np.array(state["centroids"])))
    state["centroids"], state["iteration"] = new_centers, state["iteration"] + 1
    state["history"].append({"iter": state["iteration"], "movement": movement, "wcss": state.get("current_wcss", 0)})
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id][f"13_Update_Centroid_{state['iteration']}"] = pd.DataFrame(new_centers)
    add_to_checklist(x_session_id, f"Centroid Update #{state['iteration']}")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "new_centroids": new_centers, "iteration": state["iteration"], "movement": movement, "sample_work": {"explanation": "Centroid baru dihitung dari rata-rata anggota cluster."}}

@app.post("/stepwise/check-convergence/")
async def check_convergence(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    state = session["algo_state"]
    is_conv = state["history"][-1]["movement"] < 1e-4 if state["history"] else False
    eval_res = {}
    if is_conv:
        eval_res = calculate_cluster_metrics(session["df"], state["features"], np.array(state["assignments"]), state["k"], session["config"].get("ahp_weights"))
        session.update({"metrics": eval_res, "all_results": {session["config"].get("mode", "kmeans"): eval_res}})
        ensure_audit(x_session_id)
        audit_checkpoints[x_session_id]["14_Stabilitas_Konvergensi"] = pd.DataFrame(state["history"])
        add_to_checklist(x_session_id, "Convergence Reached")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "is_converged": is_conv, "iteration": state["iteration"], "history": state["history"], "centroids": state["centroids"], "evaluation": eval_res}

# --- 7. FCM LOGIC ---

@app.post("/stepwise/fcm-init/")
async def fcm_init_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3, "m": 2.0})):
    session = await get_session(x_session_id)
    k, m = params.get("k", 3), params.get("m", 2.0)
    feats = session["config"].get("features", list(session["df"].columns))
    X = get_weighted_x(session["df"][feats].fillna(0).values, session["config"].get("ahp_weights"), feats)
    U = np.random.dirichlet(np.ones(k), size=len(X)).T
    session["algo_state"] = {"mode": "fcm", "iteration": 0, "U": U.tolist(), "X": X.tolist(), "features": feats, "k": k, "m": m, "history": []}
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id]["10_Inisialisasi_FCM_U"] = pd.DataFrame(U.T)
    add_to_checklist(x_session_id, "Inisialisasi FCM")
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/stepwise/fcm-calculate-centers/")
async def fcm_calc_centers_step(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    state = session["algo_state"]
    X, U, m = np.array(state["X"]), np.array(state["U"]), state["m"]
    centers = ((U**m) @ X) / ((U**m).sum(axis=1)[:, np.newaxis] + 1e-10)
    state["centroids"] = centers.tolist()
    ensure_audit(x_session_id)
    audit_checkpoints[x_session_id]["13_Pusat_FCM"] = pd.DataFrame(centers)
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "centroids": centers.tolist()}

@app.post("/stepwise/fcm-update-membership/")
async def fcm_update_u_step(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    state = session["algo_state"]
    X, U_old, centers, m = np.array(state["X"]), np.array(state["U"]), np.array(state["centroids"]), state["m"]
    dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
    new_U = ( (dists ** (-2.0 / (m - 1))) / (dists ** (-2.0 / (m - 1))).sum(axis=1)[:, np.newaxis] ).T
    diff = np.linalg.norm(new_U - U_old)
    state["U"], state["iteration"] = new_U.tolist(), state["iteration"] + 1
    state["history"].append({"iter": state["iteration"], "diff": float(diff)})
    add_to_checklist(x_session_id, "Optimasi Keanggotaan")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "diff": float(diff)}

@app.post("/stepwise/fcm-iteration/")
async def fcm_iteration_step(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    state = session["algo_state"]
    X, U, m, k = np.array(state["X"]), np.array(state["U"]), state["m"], state["k"]
    centers = ((U**m) @ X) / ((U**m).sum(axis=1)[:, np.newaxis] + 1e-10)
    dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
    new_U = ( (dists ** (-2.0 / (m - 1))) / (dists ** (-2.0 / (m - 1))).sum(axis=1)[:, np.newaxis] ).T
    diff = np.linalg.norm(new_U - U)
    state.update({"U": new_U.tolist(), "centroids": centers.tolist(), "iteration": state["iteration"] + 1})
    state["history"].append({"iter": state["iteration"], "diff": float(diff)})
    if diff < 1e-4:
        eval_f = calculate_cluster_metrics(session["df"], state["features"], np.argmax(new_U, axis=0), k, session["config"].get("ahp_weights"))
        session.update({"metrics": eval_f, "all_results": {"fcm": eval_f}})
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "is_converged": diff < 1e-4, "diff": float(diff)}

@app.post("/stepwise/auto-converge/")
async def auto_converge(x_session_id: Optional[str] = Header(None)):
    """Step 20: Full Algorithm Automation with Convergence Trail (Riset Standard)."""
    session = await get_session(x_session_id)
    state = session["algo_state"]
    k, feats = state["k"], state["features"]
    ahp = session["config"].get("ahp_weights")
    X = get_weighted_x(session["df"][feats].fillna(0).values, ahp, feats)

    history = []

    # --- 1. FCM AUTO CONVERGE ---
    if state.get("mode") == "fcm":
        U, m = np.array(state["U"]), state["m"]
        for i in range(1, 101):
            U_m = U ** m
            centers = (U_m @ X) / (U_m.sum(axis=1)[:, np.newaxis] + 1e-10)
            dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
            new_U = ( (dists ** (-2.0 / (m - 1))) / (dists ** (-2.0 / (m - 1))).sum(axis=1)[:, np.newaxis] ).T
            diff = float(np.linalg.norm(new_U - U))

            # WCSS equivalent for FCM (J_m)
            jm = float(np.sum((U_m).T * (dists**2)))

            history.append({"iter": i, "movement": diff, "wcss": jm})
            U = new_U
            if diff < 1e-4: break

        eval_f = calculate_cluster_metrics(session["df"], feats, np.argmax(U, axis=0), k, ahp)
        session.update({
            "metrics": eval_f,
            "all_results": {"fcm": eval_f},
            "algo_state": {**state, "U": U.tolist(), "centroids": centers.tolist(), "is_converged": True, "history": history}
        })
        add_to_checklist(x_session_id, "Riset Selesai (Auto FCM)")
        sync_session_to_firebase(x_session_id)
        return {"status": "success", "is_converged": True, "evaluation": eval_f, "centroids": centers.tolist(), "iteration": len(history), "history": history}

    # --- 2. K-MEANS AUTO CONVERGE ---
    else:
        centroids = np.array(state["centroids"])
        for i in range(1, 101):
            # Assignment
            dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            assignments = np.argmin(dists, axis=1)
            wcss = float(np.sum(np.min(dists, axis=1)**2))

            # Update
            new_centroids = np.array([X[assignments == j].mean(axis=0) if len(X[assignments == j]) > 0 else centroids[j] for j in range(k)])
            movement = float(np.linalg.norm(new_centroids - centroids))

            history.append({"iter": i, "movement": movement, "wcss": wcss})
            centroids = new_centroids
            if movement < 1e-4: break

        eval_k = calculate_cluster_metrics(session["df"], feats, assignments, k, ahp)
        session.update({
            "metrics": eval_k,
            "all_results": {"kmeans": eval_k},
            "algo_state": {**state, "centroids": centroids.tolist(), "assignments": assignments.tolist(), "is_converged": True, "history": history}
        })
        add_to_checklist(x_session_id, "Riset Selesai (Auto K-Means)")
        sync_session_to_firebase(x_session_id)
        return {"status": "success", "is_converged": True, "iteration": len(history), "centroids": centroids.tolist(), "evaluation": eval_k, "history": history}

# --- 8. BENCHMARK & ANALYSIS ---

@app.get("/stepwise/wcss-detail/")
async def get_wcss_detail(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    return {"status": "success", "wcss": session.get("metrics", {}).get("wcss", 0.0)}

@app.get("/stepwise/dbi-detail/")
async def get_dbi_detail(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    return {"status": "success", "dbi": session.get("metrics", {}).get("dbi", 0.0)}

@app.post("/stepwise/benchmark/")
@app.post("/stepwise/compare-all/")
async def stepwise_benchmark(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    X = get_weighted_x(session["df"][session["config"]["features"]].fillna(0).values, session["config"].get("ahp_weights"), session["config"]["features"])
    k = session["config"].get("k", 3)
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    km_metrics = calculate_cluster_metrics(session["df"], session["config"]["features"], km.labels_, k, session["config"].get("ahp_weights"))
    U = np.random.dirichlet(np.ones(k), size=len(X)).T
    m = session["config"].get("m", 2.0)
    for _ in range(30):
        centers = ((U**m) @ X) / ((U**m).sum(axis=1)[:, np.newaxis] + 1e-10)
        dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
        U = ( (dists ** (-2.0 / (m - 1))) / (dists ** (-2.0 / (m - 1))).sum(axis=1)[:, np.newaxis] ).T
    fcm_metrics = calculate_cluster_metrics(session["df"], session["config"]["features"], np.argmax(U, axis=0), k, session["config"].get("ahp_weights"))
    sig = perform_significance_test(X, km.labels_, np.argmax(U, axis=0))
    return {"status": "success", "comparison_data": {"kmeans": km_metrics, "fcm": fcm_metrics}, "significance": sig}

@app.get("/stepwise/final-analysis/")
async def get_final_analysis(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    return {"status": "success", "config": session.get("config", {}), "metrics": session.get("metrics", {}), "hasil_cluster": session["df"].fillna(0).to_dict(orient="records")}

@app.get("/stepwise/explain-siswa/")
async def explain_siswa(x_session_id: Optional[str] = Header(None), nis: str = ""):
    session = await get_session(x_session_id)
    student = session["df"][session["df"]["nis"] == nis]
    if student.empty: raise HTTPException(404, "Not found")
    idx = int(student["cluster"].values[0])
    target = np.array(session["metrics"]["centroids"])[idx]
    X_s = student[session["config"]["features"]].fillna(0).values[0]
    ranges = np.ptp(session["df"][session["config"]["features"]].fillna(0).values, axis=0) + 1e-10
    contribs = [{"feature": f, "val": 1.0 - (abs(X_s[i] - target[i]) / ranges[i])} for i, f in enumerate(session["config"]["features"])]
    return {"status": "success", "contributions": contribs}

# --- 9. HOUSEKEEPING ---

@app.post("/stepwise/save-to-history/")
async def save_to_history(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.get("/stepwise/export-pdf/")
async def export_pdf_route(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    pdf = ResearchReportPDF()
    pdf.add_page()
    pdf.chapter_title("LAPORAN RISET")
    return StreamingResponse(io.BytesIO(pdf.output()), media_type="application/pdf")

@app.get("/stepwise/export-excel/")
async def export_excel_route(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in audit_checkpoints.get(x_session_id, {}).items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.get("/stepwise/build-manuscript/")
async def export_docx_route(x_session_id: Optional[str] = Header(None)):
    session = await get_session(x_session_id)
    buf = generate_manuscript_docx(session, x_session_id, get_weighted_x)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
