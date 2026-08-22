from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Body
from fastapi.responses import StreamingResponse
import os
import sys
import pandas as pd
import numpy as np
import time
import io
import uuid
from typing import Optional, List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --- LOCAL MODULES ---
from api.utils import (
    sessions, audit_checkpoints, db, ensure_session,
    sync_session_to_firebase, add_to_checklist, get_representative_data
)
from api.statistics import (
    calculate_cluster_metrics, calculate_xie_beni, calculate_partition_entropy,
    calculate_hopkins, calculate_ahp_weights_and_cr, get_weighted_x
)
from api.reports import (
    ResearchReportPDF, generate_radar_chart_bytes, generate_bar_chart_bytes,
    generate_silhouette_chart_bytes, generate_manuscript_docx
)

app = FastAPI(title="SIMORBATAS Python AI Runtime (Vercel Modular)", version="2.0.0")

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
        initial_preview = get_representative_data(df)

        sessions[x_session_id] = {
            "df": df, "filename": file.filename, "config": {"filename": file.filename},
            "metrics": {}, "all_results": {}, "checkpoints": {"Data Asli": initial_preview},
            "audit": {"initial_rows": len(df), "initial_cols": len(df.columns), "missing_before": int(df.isnull().sum().sum()), "outliers_removed": 0, "normalization_method": "None", "execution_checklist": []}
        }

        if x_session_id not in audit_checkpoints: audit_checkpoints[x_session_id] = {}
        audit_checkpoints[x_session_id]["01_Data_Asli"] = df.copy()

        sync_session_to_firebase(x_session_id)
        return {"status": "success", "jumlah_data": len(df), "columns": list(df.columns), "session_id": x_session_id}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/stepwise/raw-data/")
async def get_raw_data(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    data_tampil = get_representative_data(df)
    return {
        "columns": list(df.columns), "total_rows": int(len(df)),
        "data": pd.DataFrame(data_tampil).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records"),
        "is_representative": True, "note": "Menampilkan 3 data pertama dan 2 data terakhir."
    }

@app.post("/stepwise/cleaning/")
async def stepwise_cleaning(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    sessions[x_session_id]["checkpoints"]["Pembersihan Data (Sebelum)"] = get_representative_data(df)

    df = df.dropna(how='all').dropna(axis=1, how='all').drop_duplicates()
    for col in df.select_dtypes(include=['object']).columns: df[col] = df[col].astype(str).str.strip()
    sessions[x_session_id]["df"] = df
    sessions[x_session_id]["checkpoints"]["Pembersihan Data (Sesudah)"] = get_representative_data(df)

    if x_session_id not in audit_checkpoints: audit_checkpoints[x_session_id] = {}
    audit_checkpoints[x_session_id]["04_Data_Cleaning"] = df.copy()

    add_to_checklist(x_session_id, "Pembersihan Data")
    return {"status": "success", "final_rows": len(df), "log": "Cleaning selesai."}

@app.post("/stepwise/missing-value/")
async def stepwise_missing(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    sessions[x_session_id]["checkpoints"]["Imputasi Nilai Kosong (Sebelum)"] = get_representative_data(df)
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols: df[col] = df[col].fillna(df[col].median())
    sessions[x_session_id]["df"] = df
    sessions[x_session_id]["checkpoints"]["Imputasi Nilai Kosong (Sesudah)"] = get_representative_data(df)

    if x_session_id not in audit_checkpoints: audit_checkpoints[x_session_id] = {}
    audit_checkpoints[x_session_id]["05_Imputasi_Data"] = df.copy()

    add_to_checklist(x_session_id, "Imputasi Data")
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

    if len(num_df) < 5: return {"status": "success", "outlier_count": 0, "message": "Data terlalu sedikit."}

    from scipy.stats import chi2
    Q1, Q3 = num_df.quantile(0.25), num_df.quantile(0.75)
    IQR = Q3 - Q1
    iqr_mask = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)
    z_scores = np.abs((num_df - num_df.mean()) / (num_df.std() + 1e-10))
    z_mask = (z_scores > 3).any(axis=1)
    m_mask = np.zeros(len(num_df), dtype=bool)
    try:
        X = num_df.values
        mu = np.mean(X, axis=0)
        cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
        inv_cov = np.linalg.inv(cov)
        diff = X - mu
        md_squared = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        threshold = chi2.ppf(0.999, df=X.shape[1])
        m_mask = md_squared > threshold
    except: pass

    outliers_mask = iqr_mask | z_mask | m_mask
    sessions[x_session_id]["checkpoints"]["Deteksi Outlier (Sesudah)"] = get_representative_data(df[~outliers_mask])
    if x_session_id not in audit_checkpoints: audit_checkpoints[x_session_id] = {}
    audit_checkpoints[x_session_id]["06_Audit_Outlier"] = df[~outliers_mask].copy()

    add_to_checklist(x_session_id, "Audit Outlier")
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
    sessions[x_session_id]["checkpoints"]["Konversi Kategorikal (Sesudah)"] = get_representative_data(df)
    if x_session_id not in audit_checkpoints: audit_checkpoints[x_session_id] = {}
    audit_checkpoints[x_session_id]["03_Konversi_Kategori"] = df.copy()

    add_to_checklist(x_session_id, "Konversi Fitur")
    return {"status": "success", "mappings": mapping_details}

@app.get("/stepwise/normalization-stats/")
async def get_norm_stats(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    num_df = df.select_dtypes(include=['number'])
    stats = {}
    for col in num_df.columns:
        series = num_df[col].dropna()
        if len(series) > 0:
            stats[col] = {"min": float(series.min()), "max": float(series.max()), "mean": float(series.mean()), "median": float(series.median()), "std": float(series.std()) if len(series) > 1 else 0.0, "variance": float(series.var()) if len(series) > 1 else 0.0}
    return {"status": "success", "stats": stats}

@app.get("/stepwise/correlation-matrix/")
async def get_correlation_analysis(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    df = sessions[x_session_id]["df"]
    config = sessions[x_session_id].get("config", {})
    features = config.get("features", list(df.select_dtypes(include=[np.number]).columns))
    num_df = df[features].select_dtypes(include=[np.number]).fillna(0)
    corr_matrix = num_df.corr()
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", center=0, square=True, annot=True, fmt=".2f")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {"status": "success", "heatmap_image": img_base64, "correlation_data": corr_matrix.to_dict()}

@app.post("/stepwise/normalization/")
async def stepwise_norm(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    from sklearn.preprocessing import MinMaxScaler
    df = sessions[x_session_id]["df"]
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        sessions[x_session_id]["df"] = df
        sessions[x_session_id]["scaler"] = scaler
        audit_checkpoints[x_session_id]["07_Penskalaan_Fitur"] = df.copy()
        add_to_checklist(x_session_id, "Normalisasi Data")
    return {"status": "success"}

@app.post("/stepwise/standardization/")
async def stepwise_standard(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    from sklearn.preprocessing import StandardScaler
    df = sessions[x_session_id]["df"]
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        sessions[x_session_id]["df"] = df
        sessions[x_session_id]["scaler"] = scaler
        audit_checkpoints[x_session_id]["07_Penskalaan_Fitur"] = df.copy()
        add_to_checklist(x_session_id, "Standardisasi Data")
    return {"status": "success"}

@app.post("/stepwise/robust-scaling/")
async def stepwise_robust_scaling(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    from sklearn.preprocessing import RobustScaler
    df = sessions[x_session_id]["df"]
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        scaler = RobustScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        sessions[x_session_id]["df"] = df
        sessions[x_session_id]["scaler"] = scaler
        audit_checkpoints[x_session_id]["07_Penskalaan_Fitur"] = df.copy()
        add_to_checklist(x_session_id, "Robust Scaling")
    return {"status": "success"}

@app.get("/stepwise/quality-report/")
async def get_quality_report(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]
    num_cols = list(df.select_dtypes(include=['number']).columns)
    hopkins = calculate_hopkins(df[num_cols].fillna(0).values) if len(num_cols) >= 2 else 0.5
    return {"status": "success", "rows": len(df), "numeric_features": len(num_cols), "hopkins_statistic": hopkins, "execution_checklist": sessions[x_session_id]["audit"].get("execution_checklist", [])}

@app.get("/stepwise/checkpoints/")
async def get_checkpoints(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    return {"status": "success", "checkpoints": sessions[x_session_id].get("checkpoints", {})}

@app.get("/stepwise/universal-dataset/")
async def get_universal_dataset(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    df = sessions[x_session_id]["df"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return {"status": "success", "columns": list(df.columns), "data": df.head(500).to_dict(orient="records")}

@app.get("/stepwise/session-state/")
async def get_session_state(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    return {"state": "UPLOADED" if x_session_id in sessions else "IDLE"}

@app.post("/stepwise/elbow/")
async def stepwise_elbow(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    X = sessions[x_session_id]["df"].select_dtypes(include=[np.number]).fillna(0)
    wcss = [{"k": i, "wcss": float(KMeans(n_clusters=i, n_init=10, random_state=42).fit(X).inertia_)} for i in range(1, 11)]
    add_to_checklist(x_session_id, "Analisis Elbow")
    return {"status": "success", "data": wcss}

@app.post("/stepwise/gap-statistic/")
async def stepwise_gap_statistic(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    df = sessions[x_session_id]["df"]
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=[np.number]).columns))
    X = df[features].select_dtypes(include=[np.number]).fillna(0).values
    if len(X) < 10: return {"status": "success", "recommended_k": 3}
    n_samples, n_features = X.shape
    ks = range(1, 7)
    gaps = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X)
        log_wcss = np.log(km.inertia_ + 1e-10)
        ref_log_wcss = []
        for i in range(5):
            rand = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(n_samples, n_features))
            ref_log_wcss.append(np.log(KMeans(n_clusters=k, n_init=5, random_state=i).fit(rand).inertia_ + 1e-10))
        gaps.append({"k": k, "gap": float(np.mean(ref_log_wcss) - log_wcss)})
    recommended_k = int(ks[np.argmax([g["gap"] for g in gaps])])
    add_to_checklist(x_session_id, "Gap Statistic")
    return {"status": "success", "gap_values": gaps, "recommended_k": recommended_k}

@app.get("/stepwise/compare_k/")
async def stepwise_compare_k(x_session_id: Optional[str] = Header(None)):
    """
    S2 OPTIMIZATION: Multi-Metric Cluster Optimization (K=2 to K=10).
    Calculates Silhouette and DBI for multiple K values to find the mathematical 'Sweet Spot'.
    """
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[x_session_id]["df"]
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=[np.number]).columns))
    X_raw = df[features].select_dtypes(include=[np.number]).fillna(0).values

    # Use existing weights if available
    ahp_weights = sessions[x_session_id]["config"].get("ahp_weights")
    X = get_weighted_x(X_raw, ahp_weights, features)

    if len(X) < 10:
        return {"status": "error", "message": "Dataset terlalu kecil untuk optimasi K."}

    results = []
    for k in range(2, 11):
        # Using KMeans++ as the standard optimization baseline
        km = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=42).fit(X)
        labels = km.labels_

        sil = float(silhouette_score(X, labels))
        dbi = float(davies_bouldin_score(X, labels))
        chi = float(calinski_harabasz_score(X, labels))

        results.append({
            "k": k,
            "silhouette": sil,
            "dbi": dbi,
            "chi": chi,
            "wcss": float(km.inertia_)
        })

    # Heuristic for Best K (Max Silhouette and Min DBI)
    best_k_sil = max(results, key=lambda x: x["silhouette"])["k"]
    best_k_dbi = min(results, key=lambda x: x["dbi"])["k"]

    add_to_checklist(x_session_id, "Optimasi Jumlah K")
    sync_session_to_firebase(x_session_id)

    return {
        "status": "success",
        "results": results,
        "best_k_silhouette": best_k_sil,
        "best_k_dbi": best_k_dbi,
        "interpretation": f"Berdasarkan validasi internal, K={best_k_sil} memiliki kepadatan terbaik (Silhouette), sedangkan K={best_k_dbi} memiliki pemisahan terbaik (DBI)."
    }

# --- K-MEANS STEP-BY-STEP LOGIC ---

@app.post("/stepwise/init-centroids/")
async def init_centroids_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    """
    TAHAP 1 K-MEANS: Inisialisasi Pusat Massa (Centroids).
    Memilih k titik secara acak dari dataset sebagai titik awal kelompok.
    """
    await ensure_session(x_session_id)
    df, k = sessions[x_session_id]["df"], params.get("k", 3)
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=[np.number]).columns))
    num_df = df[features].select_dtypes(include=[np.number]).fillna(0)
    centroids = num_df.sample(n=k, random_state=42).values.tolist()
    sessions[x_session_id]["algo_state"] = {"iteration": 0, "centroids": centroids, "features": features, "k": k, "history": [], "is_converged": False}
    audit_checkpoints[x_session_id]["10_Inisialisasi_Centroid"] = pd.DataFrame(centroids, columns=features)
    add_to_checklist(x_session_id, "Centroid Init")
    return {"status": "success", "centroids": centroids}

@app.post("/stepwise/calculate-distances/")
async def calculate_distances_step(x_session_id: Optional[str] = Header(None)):
    """
    TAHAP 2 K-MEANS: Menghitung Jarak Euclidean.
    Mengukur seberapa jauh setiap siswa dari setiap pusat kelompok (Centroid).
    Jika AHP aktif, jarak ini dikalikan dengan bobot variabel (Weighted Euclidean).
    """
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    X = sessions[x_session_id]["df"][state["features"]].fillna(0).values
    centroids = np.array(state["centroids"])
    ahp = sessions[x_session_id]["config"].get("ahp_weights")
    if ahp:
        w = np.array([ahp.get(f, 1.0) for f in state["features"]])
        dists = [np.linalg.norm(centroids * np.sqrt(w) - row * np.sqrt(w), axis=1).tolist() for row in X]
    else: dists = [np.linalg.norm(centroids - row, axis=1).tolist() for row in X]
    state["distances"] = dists
    audit_checkpoints[x_session_id]["11_Matriks_Jarak"] = pd.DataFrame(dists)
    add_to_checklist(x_session_id, "Euclidean Distance")
    return {"status": "success", "sample": dists[0]}

@app.post("/stepwise/assign-clusters/")
async def assign_clusters_step(x_session_id: Optional[str] = Header(None)):
    """
    TAHAP 3 K-MEANS: Pengelompokan (Assignment).
    Menempatkan setiap siswa ke kelompok (klaster) yang jaraknya paling dekat.
    """
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    dists = np.array(state["distances"])
    assignments = np.argmin(dists, axis=1).tolist()
    state["assignments"], state["current_wcss"] = assignments, float(np.sum(np.min(dists, axis=1)**2))
    audit_checkpoints[x_session_id]["12_Pengelompokan_Siswa"] = pd.DataFrame(assignments)
    add_to_checklist(x_session_id, "Cluster Assignment")
    return {"status": "success", "counts": len(assignments)}

@app.post("/stepwise/update-centroids/")
async def update_centroids_step(x_session_id: Optional[str] = Header(None)):
    """
    TAHAP 4 K-MEANS: Memperbarui Pusat Massa (Update).
    Menghitung ulang posisi pusat kelompok berdasarkan rata-rata posisi anggota di dalamnya.
    """
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    df, assignments = sessions[x_session_id]["df"][state["features"]].fillna(0), np.array(state["assignments"])
    new_centroids = [df[assignments == i].mean(axis=0).values.tolist() if len(df[assignments == i]) > 0 else state["centroids"][i] for i in range(state["k"])]
    movement = float(np.linalg.norm(np.array(new_centroids) - np.array(state["centroids"])))
    state["centroids"], state["iteration"] = new_centroids, state["iteration"] + 1
    state["history"].append({"iter": state["iteration"], "movement": movement})
    audit_checkpoints[x_session_id][f"13_Update_Centroid_{state['iteration']}"] = pd.DataFrame(new_centroids)
    add_to_checklist(x_session_id, f"Update Centroid #{state['iteration']}")
    return {"status": "success", "movement": movement}

@app.post("/stepwise/check-convergence/")
async def check_convergence(x_session_id: Optional[str] = Header(None)):
    """
    TAHAP 5 K-MEANS: Uji Konvergensi.
    Mengecek apakah posisi pusat kelompok masih bergeser secara signifikan.
    Jika geseran < 0.0001, algoritma dinyatakan selesai (Konvergen).
    """
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    is_converged = state["history"][-1]["movement"] < 1e-4 if state["history"] else False
    if is_converged:
        ahp = sessions[x_session_id]["config"].get("ahp_weights")
        eval = calculate_cluster_metrics(sessions[x_session_id]["df"], state["features"], np.array(state["assignments"]), state["k"], ahp)
        sessions[x_session_id].update({"metrics": eval, "all_results": {sessions[x_session_id]["config"].get("mode", "kmeans"): eval}})
        audit_checkpoints[x_session_id]["14_Stabilitas_Konvergensi"] = pd.DataFrame(state["history"])
        add_to_checklist(x_session_id, "Convergence Reached")
    return {"status": "success", "is_converged": is_converged}

# --- FUZZY C-MEANS (FCM) STEP-BY-STEP LOGIC ---

@app.post("/stepwise/fcm-init/")
async def fcm_init_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3, "m": 2.0})):
    """
    TAHAP 1 FCM: Inisialisasi Matriks Keanggotaan (U).
    Memberikan nilai probabilitas acak (0-1) bagi setiap siswa untuk masuk ke setiap klaster.
    Total nilai probabilitas satu siswa untuk semua klaster harus sama dengan 1.
    """
    await ensure_session(x_session_id)
    df = sessions[x_session_id]["df"]
    config = sessions[x_session_id].get("config", {})
    k, m = params.get("k", 3), params.get("m", 2.0)
    features = config.get("features", list(df.select_dtypes(include=[np.number]).columns))
    X = get_weighted_x(df[features].fillna(0).values, config.get("ahp_weights"), features)
    U = np.random.dirichlet(np.ones(k), size=len(X)).T
    sessions[x_session_id]["algo_state"] = {"mode": "fcm", "iteration": 0, "U": U.tolist(), "X": X.tolist(), "features": features, "k": k, "m": m, "history": [], "is_converged": False}
    for j in range(k): df[f"membership_c{j}"] = np.round(U[j, :], 4).tolist()
    audit_checkpoints[x_session_id]["10_Inisialisasi_FCM_Matrix_U"] = pd.DataFrame(U.T, columns=[f"C{i+1}" for i in range(k)])
    add_to_checklist(x_session_id, "Inisialisasi FCM")
    return {"status": "success", "message": "FCM Init OK"}

@app.post("/stepwise/fcm-calculate-centers/")
async def fcm_calc_centers_step(x_session_id: Optional[str] = Header(None)):
    """
    TAHAP 2 FCM: Menghitung Pusat Klaster Fuzzy (V).
    Menghitung koordinat pusat kelompok dengan mempertimbangkan 'derajat keanggotaan' setiap siswa.
    Siswa dengan nilai fuzzy tinggi akan lebih kuat menarik posisi pusat klaster.
    """
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    X, U, m, k = np.array(state["X"]), np.array(state["U"]), state["m"], state["k"]
    U_m = U ** m
    centers = (U_m @ X) / (U_m.sum(axis=1)[:, np.newaxis] + 1e-10)
    state["centroids"] = centers.tolist()
    audit_checkpoints[x_session_id]["13_Pembaruan_Pusat_FCM"] = pd.DataFrame(centers, columns=state["features"])
    add_to_checklist(x_session_id, "Kalkulasi Pusat V")
    return {"status": "success", "centroids": centers.tolist()}

@app.post("/stepwise/fcm-update-membership/")
async def fcm_update_u_step(x_session_id: Optional[str] = Header(None)):
    """
    TAHAP 3 FCM: Memperbarui Matriks Keanggotaan (U).
    Menghitung ulang probabilitas setiap siswa berdasarkan jarak terbaru ke pusat klaster.
    Menerapkan parameter 'm' (Fuzzifier) untuk menentukan tingkat kekaburan kelompok.
    """
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    X, U_old, centers, m, k = np.array(state["X"]), np.array(state["U"]), np.array(state["centroids"]), state["m"], state["k"]
    dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
    inv_dists = dists ** (-2.0 / (m - 1))
    new_U = (inv_dists / inv_dists.sum(axis=1)[:, np.newaxis]).T
    diff = np.linalg.norm(new_U - U_old)
    state["U"] = new_U.tolist()
    state["iteration"] += 1
    state["history"].append({"iter": state["iteration"], "diff": float(diff)})
    audit_checkpoints[x_session_id]["11_Matriks_Keanggotaan_FCM"] = pd.DataFrame(new_U.T, columns=[f"C{i+1}" for i in range(k)])
    add_to_checklist(x_session_id, "Optimasi Keanggotaan")
    return {"status": "success", "iteration": state["iteration"], "diff": float(diff)}

@app.post("/stepwise/fcm-iteration/")
async def fcm_iteration_step(x_session_id: Optional[str] = Header(None)):
    """
    LOOP UTAMA FCM: Menggabungkan Pembaruan V dan U.
    Menjalankan satu siklus penuh optimasi fuzzy hingga tercapai konvergensi.
    """
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    X, U, m, k = np.array(state["X"]), np.array(state["U"]), state["m"], state["k"]
    U_m = U ** m
    centers = (U_m @ X) / (U_m.sum(axis=1)[:, np.newaxis] + 1e-10)
    dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
    inv_dists = dists ** (-2.0 / (m - 1))
    new_U = (inv_dists / inv_dists.sum(axis=1)[:, np.newaxis]).T
    diff = np.linalg.norm(new_U - U)
    state["U"], state["centroids"], state["iteration"] = new_U.tolist(), centers.tolist(), state["iteration"] + 1
    state["history"].append({"iter": state["iteration"], "diff": float(diff)})
    audit_checkpoints[x_session_id][f"13_FCM_Update_Iter_{state['iteration']}"] = pd.DataFrame(centers, columns=state["features"])
    is_converged = diff < 1e-4
    state["is_converged"] = is_converged
    if is_converged:
        ahp = sessions[x_session_id]["config"].get("ahp_weights")
        eval = calculate_cluster_metrics(sessions[x_session_id]["df"], state["features"], np.argmax(new_U, axis=0), k, ahp)
        eval.update({"xb": calculate_xie_beni(X, new_U, centers, m), "pe": calculate_partition_entropy(new_U)})
        sessions[x_session_id].update({"metrics": eval, "all_results": {"fcm": eval}})
        add_to_checklist(x_session_id, "Riset Selesai")
    return {"status": "success", "is_converged": is_converged, "diff": float(np.nan_to_num(diff))}

@app.post("/stepwise/ahp-calculate/")
async def ahp_calculate(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body(...)):
    await ensure_session(x_session_id)
    features, matrices_raw = params.get("features"), params.get("matrices", [params.get("matrix")])
    matrices = [np.array(m) for m in matrices_raw if m]
    consensus = np.exp(np.mean(np.log(np.fmax(np.stack(matrices), 1e-10)), axis=0))
    weights, cr = calculate_ahp_weights_and_cr(consensus)
    weight_dict = {f: float(np.nan_to_num(w)) for f, w in zip(features, weights)}
    sessions[x_session_id]["config"].update({"ahp_weights": weight_dict, "ahp_cr": cr})
    audit_checkpoints[x_session_id]["08_Pembobotan_Variabel"] = pd.DataFrame(list(weight_dict.items()))
    add_to_checklist(x_session_id, "AHP Konsensus")
    return {"status": "success", "weights": weight_dict, "cr": cr}

@app.post("/stepwise/init-centroids-ga/")
async def init_centroids_ga(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    await ensure_session(x_session_id)
    df, k = sessions[x_session_id]["df"], params.get("k", 3)
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=[np.number]).columns))
    X = get_weighted_x(df[features].fillna(0).values, sessions[x_session_id]["config"].get("ahp_weights"), features)
    from sklearn.cluster import kmeans_plusplus
    best_centroids, _ = kmeans_plusplus(X, n_clusters=k, random_state=42)
    sessions[x_session_id]["algo_state"] = {"iteration": 0, "centroids": best_centroids.tolist(), "features": features, "k": k, "history": [], "is_converged": False, "method": "ga"}
    add_to_checklist(x_session_id, "Inisialisasi GA")
    return {"status": "success", "centroids": best_centroids.tolist()}

@app.post("/stepwise/auto-converge/")
async def auto_converge(x_session_id: Optional[str] = Header(None)):
    """
    OPTIMASI OTOMATIS: Menjalankan loop algoritma hingga konvergen tanpa intervensi manual.
    Digunakan untuk mempercepat riset setelah peneliti memahami proses manual.
    """
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    df, features, k = sessions[x_session_id]["df"], state["features"], state["k"]
    X = df[features].fillna(0).values
    if state.get("mode") == "fcm":
        U, m = np.array(state["U"]), state["m"]
        for i in range(100):
            U_m = U ** m
            centers = (U_m @ X) / (U_m.sum(axis=1)[:, np.newaxis] + 1e-10)
            dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
            new_U = (dists ** (-2.0 / (m - 1)) / (dists ** (-2.0 / (m - 1))).sum(axis=1)[:, np.newaxis]).T
            if np.linalg.norm(new_U - U) < 1e-4: break
            U = new_U
        eval = calculate_cluster_metrics(df, features, np.argmax(U, axis=0), k, sessions[x_session_id]["config"].get("ahp_weights"))
        sessions[x_session_id].update({"metrics": eval, "all_results": {"fcm": eval}})
        audit_checkpoints[x_session_id]["15_Hasil_Rekomendasi_SPK"] = df.copy()
        return {"status": "success", "is_converged": True}
    return {"status": "error", "msg": "K-Means auto not yet modularized"}

@app.get("/stepwise/explain-siswa/")
async def explain_siswa(x_session_id: Optional[str] = Header(None), nis: str = ""):
    await ensure_session(x_session_id)
    df, session = sessions[x_session_id]["df"], sessions[x_session_id]
    student = df[df["nis"] == nis]
    if student.empty: raise HTTPException(404, "Siswa not found")
    features = session["config"].get("features", [])
    X_student = student[features].fillna(0).values[0]
    cluster_idx = int(student["cluster"].values[0])
    target_centroid = np.array(session["metrics"]["centroids"])[cluster_idx]
    global_mean = df[features].mean(numeric_only=True).values
    ranges = np.ptp(df[features].fillna(0).values, axis=0) + 1e-10
    contributions = []
    for i, f in enumerate(features):
        proximity = 1.0 - (abs(X_student[i] - target_centroid[i]) / ranges[i])
        dev = (X_student[i] - global_mean[i]) / ranges[i]
        contributions.append({"feature": f, "val": float(proximity), "abs_val": abs(proximity), "student_vector": X_student.tolist(), "centroid_vector": target_centroid.tolist(), "feature_names": features, "deviation": float(dev)})
    contributions = sorted(contributions, key=lambda x: x["abs_val"], reverse=True)
    top = contributions[0]
    msg = f"Siswa ini masuk kelompok terutama karena '{top['feature']}' berada {'di bawah' if top['deviation'] < 0 else 'di atas'} rata-rata."
    return {"status": "success", "explanation": msg, "contributions": contributions, "student_vector": X_student.tolist(), "centroid_vector": target_centroid.tolist(), "feature_names": features}

@app.get("/stepwise/export-pdf/")
async def export_pdf_route(x_session_id: Optional[str] = Header(None), lang: str = "id"):
    await ensure_session(x_session_id)
    session = sessions[x_session_id]
    pdf = ResearchReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.chapter_title("LAPORAN RISET" if lang == "id" else "RESEARCH REPORT")
    pdf.chapter_body(f"Metode: {session['config'].get('mode', 'K-Means')}\nDataset: {session.get('filename')}")

    # Graphs
    radar = generate_radar_chart_bytes(session['metrics'].get('cluster_profiles'), session['metrics'].get('feature_names'))
    if radar: pdf.add_image_from_buf(radar, width=120)

    buf = io.BytesIO(pdf.output())
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=Laporan_{x_session_id[:8]}.pdf"})

@app.get("/stepwise/export-excel/")
async def export_excel_route(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in audit_checkpoints.get(x_session_id, {}).items():
            df.replace([np.inf, -np.inf], np.nan).fillna("-").to_excel(writer, sheet_name=name[:31], index=False)
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=Laporan_Audit.xlsx"})

@app.get("/stepwise/build-manuscript/")
async def export_docx_route(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    buf = generate_manuscript_docx(sessions[x_session_id], x_session_id, get_weighted_x)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=Manuscript.docx"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
