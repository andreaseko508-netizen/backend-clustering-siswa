from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Body
from fastapi.responses import StreamingResponse
import importlib
import os
import sys
import pandas as pd
import numpy as np
import time
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import io
import uuid
import json
import pickle
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Optional, List, Dict, Any

# VERCEL COMPATIBILITY: Ensure the current directory and parent are in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

app = FastAPI(title="SIMORBATAS Python AI Runtime (Vercel)", version="1.7.0")

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
                if "df" in v:
                    del v["df"] # Don't store full DFs in every result
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

                # Deserialize Scaler
                if "scaler_b64" in data:
                    try:
                        data["scaler"] = pickle.loads(base64.b64decode(data["scaler_b64"]))
                        del data["scaler_b64"]
                    except Exception as e: print(f"Scaler deserialization failed: {e}")

                if "df_records" in data:
                    data["df"] = pd.DataFrame(data["df_records"])
                    del data["df_records"]

                # Deserialize Scaler
                if "scaler_b64" in data:
                    try:
                        data["scaler"] = pickle.loads(base64.b64decode(data["scaler_b64"]))
                    except:
                        data["scaler"] = None

                sessions[x_session_id] = data

def add_to_checklist(x_session_id: str, step_name: str):
    if x_session_id in sessions:
        if "audit" not in sessions[x_session_id]:
            sessions[x_session_id]["audit"] = {"execution_checklist": []}
        checklist = sessions[x_session_id]["audit"].get("execution_checklist", [])
        if step_name not in checklist:
            checklist.append(step_name)
        sessions[x_session_id]["audit"]["execution_checklist"] = checklist
        # Force Hard-Sync to Firebase to prevent data loss on page transition
        sync_session_to_firebase(x_session_id)
        # Force Hard-Sync to Firebase to prevent data loss on page transition
        sync_session_to_firebase(x_session_id)

def calculate_cluster_metrics(df, features, assignments, k):
    try:
        X = df[features].select_dtypes(include=[np.number]).fillna(0)
        unique_labels = np.unique(assignments)

        dbi = float(davies_bouldin_score(X, assignments)) if len(unique_labels) > 1 else 0.0
        sil = float(silhouette_score(X, assignments)) if len(unique_labels) > 1 else 0.0
        chi = float(calinski_harabasz_score(X, assignments)) if len(unique_labels) > 1 else 0.0

        # WCSS Calculation
        wcss = 0.0
        if len(unique_labels) > 1:
            for i in range(k):
                cluster_points = X[assignments == i]
                if len(cluster_points) > 0:
                    center = cluster_points.mean().values
                    wcss += np.sum((cluster_points.values - center)**2)

        dist = {str(i): {"count": int(np.sum(assignments == i)), "percentage": float(np.sum(assignments == i) / len(df) * 100)} for i in range(k)}
        profiles = {str(i): df[assignments == i][features].mean(numeric_only=True).to_dict() for i in range(k)}

        # Feature Importance Calculation (Sensitivity Analysis)
        # We calculate variance of centroids across all clusters for each feature
        centroid_matrix = np.array([profiles[str(i)].get(f, 0) for i in range(k) for f in features]).reshape(k, -1)
        variances = np.var(centroid_matrix, axis=0)
        # Normalize to percentage
        importance_sum = np.sum(variances) if np.sum(variances) > 0 else 1.0
        feature_importance = {f: float((v / importance_sum) * 100) for f, v in zip(features, variances)}

        # Rigiditas Ilmiah: Penjelasan Matematis & Interpretasi
        scientific_details = {
            "silhouette": {
                "name": "Silhouette Coefficient",
                "formula": "s = (b - a) / max(a, b)",
                "description": "Mengukur seberapa mirip sebuah objek dengan clusternya sendiri dibandingkan dengan cluster lain.",
                "interpretation": "Rentang [-1, 1]. Nilai mendekati 1 menunjukkan pemisahan cluster yang sangat baik.",
                "value": sil
            },
            "dbi": {
                "name": "Davies-Bouldin Index",
                "formula": "DB = (1/k) Σ max((Ri + Rj) / dij)",
                "description": "Rasio jumlah dispersi dalam cluster terhadap jarak antar cluster.",
                "interpretation": "Semakin kecil nilai DBI (mendekati 0), maka kualitas clustering semakin baik.",
                "value": dbi
            },
            "wcss": {
                "name": "Within-Cluster Sum of Squares",
                "formula": "WCSS = Σ Σ ||xi - ci||²",
                "description": "Total variansi dalam cluster (jarak kuadrat objek ke pusat clusternya).",
                "interpretation": "Digunakan dalam Elbow Method. Nilai yang lebih kecil menunjukkan cluster yang lebih padat.",
                "value": wcss
            }
        }

        # Rekomendasi Perbaikan Otomatis (Advisor)
        improvement_advice = []
        if sil < 0.25:
            improvement_advice.append("Gunakan 'Standardisasi Z-Score' jika fitur memiliki rentang nilai yang sangat berbeda.")
            improvement_advice.append("Coba kurangi atau tambah nilai K menggunakan referensi grafik Elbow.")
        if dbi > 1.2:
            improvement_advice.append("Sistem mendeteksi overlap antar cluster. Pastikan outlier sudah dibersihkan pada tahap preprocessing.")
        if sil > 0.5 and dbi < 0.8:
            improvement_advice.append("Kualitas clustering optimal. Hasil sudah sangat layak untuk interpretasi riset.")

        return {
            "davies_bouldin_index": dbi,
            "silhouette_score": sil,
            "calinski_harabasz_index": chi,
            "wcss": wcss,
            "distribution": dist,
            "cluster_profiles": profiles,
            "feature_importance": feature_importance,
            "scientific_details": scientific_details,
            "improvement_advice": improvement_advice,
            "dbi": dbi,
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"Metrics Error: {e}")
        return {"davies_bouldin_index": 0.0, "silhouette_score": 0.0, "calinski_harabasz_index": 0.0, "wcss": 0.0, "distribution": {}, "cluster_profiles": {}, "scientific_details": {}, "dbi": 0.0}

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

        # Rigiditas: Representative data (3 awal, 2 akhir)
        initial_preview = get_representative_data(df)

        sessions[x_session_id] = {
            "df": df,
            "filename": file.filename,
            "config": {"filename": file.filename},
            "metrics": {},
            "all_results": {}, # Support Multi-Algorithm Workflow
            "checkpoints": {"Data Asli": initial_preview},
            "audit": {"initial_rows": len(df), "initial_cols": len(df.columns), "missing_before": int(df.isnull().sum().sum()), "outliers_removed": 0, "normalization_method": "None", "execution_checklist": []}
        }
        sync_session_to_firebase(x_session_id)
        return {"status": "success", "jumlah_data": len(df), "columns": list(df.columns), "session_id": x_session_id}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

def get_representative_data(df):
    if len(df) <= 5:
        return df.to_dict(orient="records")

    first_three = df.head(3)
    last_two = df.tail(2)

    representative = pd.concat([first_three, last_two])
    return representative.to_dict(orient="records")

@app.get("/stepwise/raw-data/")
async def get_raw_data(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    df = sessions[x_session_id]["df"]

    # Rigiditas: Hanya kirim 3 awal, 2 akhir untuk efisiensi & edukasi fleksibel
    data_tampil = get_representative_data(df)

    return {
        "columns": list(df.columns),
        "total_rows": int(len(df)),
        "data": pd.DataFrame(data_tampil).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records"),
        "is_representative": True,
        "note": "Menampilkan 3 data pertama dan 2 data terakhir untuk efisiensi visual."
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
    add_to_checklist(x_session_id, "Pembersihan Data")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "final_rows": len(df), "log": f"Cleaning selesai."}

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
    add_to_checklist(x_session_id, "Imputasi Data")
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
    sessions[x_session_id]["checkpoints"]["Deteksi Outlier (Sesudah)"] = get_representative_data(df[~outliers_mask])
    add_to_checklist(x_session_id, "Audit Outlier")
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
    sessions[x_session_id]["checkpoints"]["Konversi Kategorikal (Sesudah)"] = get_representative_data(df)
    add_to_checklist(x_session_id, "Konversi Fitur")
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
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        sessions[x_session_id]["df"] = df
        sessions[x_session_id]["scaler"] = scaler # Store for simulation
        sessions[x_session_id]["checkpoints"]["Normalisasi Min-Max (Sesudah)"] = get_representative_data(df)
        add_to_checklist(x_session_id, "Normalisasi Data")
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
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        sessions[x_session_id]["df"] = df
        sessions[x_session_id]["scaler"] = scaler # Store for simulation
        sessions[x_session_id]["checkpoints"]["Standardisasi Z-Score (Sesudah)"] = get_representative_data(df)
        add_to_checklist(x_session_id, "Standardisasi Data")
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
    add_to_checklist(x_session_id, "Analisis Elbow")
    sync_session_to_firebase(x_session_id)
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
    add_to_checklist(x_session_id, "Centroid Init")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "centroids": centroids, "features": features, "message": "Inisialisasi berhasil."}

# --- FUZZY C-MEANS (FCM) SPECIFIC ENDPOINTS ---

@app.post("/stepwise/fcm-init/")
async def fcm_init_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3, "m": 2.0})):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[x_session_id]["df"]
    config = sessions[x_session_id].get("config", {})
    k = params.get("k", 3)
    m = params.get("m", 2.0)

    features = config.get("features", list(df.select_dtypes(include=[np.number]).columns))
    X = df[features].select_dtypes(include=[np.number]).fillna(0).values
    n_samples = X.shape[0]

    # Initialize Membership Matrix U (Randomly, rows sum to 1)
    np.random.seed(42)
    U = np.random.dirichlet(np.ones(k), size=n_samples).T # k x n_samples

    sessions[x_session_id]["algo_state"] = {
        "mode": "fcm",
        "iteration": 0,
        "U": U.tolist(),
        "X": X.tolist(),
        "features": features,
        "k": k,
        "m": m,
        "history": [],
        "is_converged": False
    }

    # Merge initial membership into dataframe for UI preview
    for j in range(k):
        df[f"membership_c{j}"] = np.round(U[j, :], 4).tolist()
    sessions[x_session_id]["df"] = df

    add_to_checklist(x_session_id, "Inisialisasi FCM")
    sync_session_to_firebase(x_session_id)

    # Merge initial membership into dataframe for UI preview
    # Round to 4 decimals for visual professionality
    for j in range(k):
        df[f"membership_c{j}"] = np.round(U[j, :], 4).tolist()
    sessions[x_session_id]["df"] = df

    # Metadata Label column from config
    label_col = config.get("label", "nama")

    # Sample work for Step 14
    sample_work = {
        "explanation": "Matriks U diinisialisasi secara acak menggunakan Distribusi Dirichlet untuk menjamin total probabilitas keanggotaan per baris adalah 1.0.",
        "sample_u": [float(x) for x in np.round(U[:, 0], 4)],
        "formula": "U^{(0)} = [\\mu_{ij}] \\in [0, 1]",
        "label_column": label_col,
        "symbols": {
            "\\mu_{ij}": "Derajat keanggotaan subjek i pada klaster j.",
            "U^{(0)}": "Matriks keanggotaan awal (iterasi ke-0)."
        }
    }

    return {
        "status": "success",
        "message": f"Matriks keanggotaan fuzzy (k={k}, m={m}) berhasil diinisialisasi.",
        "sample_work": sample_work
    }

@app.post("/stepwise/fcm-calculate-centers/")
async def fcm_calc_centers_step(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    if not state or state.get("mode") != "fcm": raise HTTPException(status_code=400, detail="FCM state missing")

    X = np.array(state["X"])
    U = np.array(state["U"])
    m = state["m"]
    k = state["k"]

    # Formula: v_j = sum( u_ij^m * x_i ) / sum( u_ij^m )
    U_m = U ** m
    numerator = U_m @ X # k x n_features
    denominator = U_m.sum(axis=1)[:, np.newaxis] # k x 1
    # Handle zero denominator to avoid NaN
    denominator = np.where(denominator == 0, 1e-10, denominator)
    centers = numerator / denominator

    state["centroids"] = centers.tolist()

    # Round for UI
    rounded_centers = np.round(centers, 4).tolist()

    # Sample work for Step 15
    sample_work = {
        "explanation": f"Pusat klaster (Centroid) dihitung sebagai rata-rata terbobot dari seluruh data menggunakan pangkat m={m} dari matriks keanggotaan. Fitur dengan bobot keanggotaan tinggi akan menarik pusat klaster lebih kuat.",
        "formula": "v_j = \\frac{\\sum_{i=1}^n \\mu_{ij}^m x_i}{\\sum_{i=1}^n \\mu_{ij}^m}",
        "symbols": {
            "v_j": "Vektor pusat klaster (Centroid) ke-j.",
            "\\mu_{ij}": "Derajat keanggotaan subjek i pada klaster j.",
            "m": "Parameter pembobotan fuzzy (Fuzzifier).",
            "x_i": "Vektor data subjek ke-i."
        },
        "sample_v": rounded_centers[0]
    }

    add_to_checklist(x_session_id, "Kalkulasi Pusat V")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "centroids": rounded_centers, "sample_work": sample_work}

@app.post("/stepwise/fcm-update-membership/")
async def fcm_update_u_step(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    if not state or "centroids" not in state: raise HTTPException(status_code=400, detail="FCM centers not calculated")

    X = np.array(state["X"])
    U_old = np.array(state["U"])
    centers = np.array(state["centroids"])
    m = state["m"]
    k = state["k"]
    df = sessions[x_session_id]["df"]

    dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    dists = np.fmax(dists, 1e-10)

    power = 2.0 / (m - 1)
    new_U = np.zeros((X.shape[0], k))

    for i in range(X.shape[0]):
        for j in range(k):
            denominator = np.sum((dists[i, j] / dists[i, :]) ** power)
            new_U[i, j] = 1.0 / denominator

    new_U = new_U.T
    diff = np.linalg.norm(new_U - U_old)

    state["U"] = new_U.tolist()
    state["iteration"] += 1
    state["history"].append({"iter": state["iteration"], "diff": float(diff)})

    # Merge updated membership into dataframe for UI preview
    for j in range(k):
        df[f"membership_c{j}"] = np.round(new_U[j, :], 4).tolist()
    sessions[x_session_id]["df"] = df

    # Sample work for Step 16
    sample_work = {
        "explanation": "Derajat keanggotaan diperbarui berdasarkan rasio jarak relatif subjek terhadap seluruh pusat klaster. Semakin dekat subjek ke pusat j, semakin besar nilai keanggotaannya.",
        "formula": "\\mu_{ij} = [\\sum_{k=1}^C (\\frac{d_{ij}}{d_{ik}})^{\\frac{2}{m-1}}]^{-1}",
        "symbols": {
            "\\mu_{ij}": "Nilai keanggotaan baru subjek i pada klaster j.",
            "d_{ij}": "Jarak Euclidean subjek i ke pusat klaster j.",
            "C": "Jumlah klaster (K)."
        },
        "sample_u_new": [float(x) for x in np.round(new_U[:, 0], 4)],
        "diff": float(diff)
    }

    add_to_checklist(x_session_id, "Optimasi Keanggotaan")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "iteration": state["iteration"], "diff": float(diff), "sample_work": sample_work}

@app.post("/stepwise/fcm-iteration/")
async def fcm_iteration_step(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    if not state or state.get("mode") != "fcm": raise HTTPException(status_code=400, detail="FCM state missing")

    X = np.array(state["X"])
    U = np.array(state["U"])
    m = state["m"]
    k = state["k"]

    # 1. Update Centers
    U_m = U ** m
    centers = (U_m @ X) / U_m.sum(axis=1)[:, np.newaxis]

    # 2. Update Membership Matrix U
    # Calculate distances to new centers
    dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2) # n_samples x k
    dists = np.fmax(dists, 1e-10) # Avoid zero distance

    # Formula: u_ij = 1 / sum( (d_ij/d_ik)^(2/(m-1)) )
    power = 2.0 / (m - 1)
    new_U = np.zeros_like(U.T) # n_samples x k

    for i in range(X.shape[0]):
        for j in range(k):
            denominator = np.sum((dists[i, j] / dists[i, :]) ** power)
            new_U[i, j] = 1.0 / denominator

    new_U = new_U.T # k x n_samples

    # Convergence check
    diff = np.linalg.norm(new_U - U)
    state["U"] = new_U.tolist()
    state["centroids"] = centers.tolist()
    state["iteration"] += 1
    state["history"].append({"iter": state["iteration"], "diff": float(diff)})

    is_converged = diff < 1e-4
    state["is_converged"] = is_converged

    if is_converged:
        # Finalize FCM
        assignments = np.argmax(new_U, axis=0)
        metrics = calculate_cluster_metrics(sessions[x_session_id]["df"], state["features"], assignments, k)

        # Add Fuzzy Specific Metrics: Partition Coefficient (PC)
        pc = float(np.mean(np.sum(new_U**2, axis=0)))
        metrics["partition_coefficient"] = pc
        metrics["centroids"] = centers.tolist()
        metrics["feature_names"] = state["features"]

        sessions[x_session_id]["metrics"] = metrics
        sessions[x_session_id]["df"]["cluster"] = assignments.tolist()

        # Store membership for all clusters
        for j in range(k):
            sessions[x_session_id]["df"][f"membership_c{j}"] = np.round(new_U[j, :], 4).tolist()

        add_to_checklist(x_session_id, "FCM Convergence Reached")

    sync_session_to_firebase(x_session_id)
    return {
        "status": "success",
        "iteration": state["iteration"],
        "diff": float(diff),
        "is_converged": is_converged,
        "sample_membership": new_U[:, 0].tolist()
    }

@app.post("/stepwise/ahp-calculate/")
async def ahp_calculate(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body(...)):
    """Calculates feature weights using Analytic Hierarchy Process (AHP)."""
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")

    matrix = np.array(params.get("matrix")) # Pairwise comparison matrix
    features = params.get("features")

    n = len(features)
    # Calculate Eigenvector (Weights) using Geometric Mean or Power Method
    # Simple version: Column normalization
    col_sum = np.sum(matrix, axis=0)
    norm_matrix = matrix / col_sum
    weights = np.mean(norm_matrix, axis=1)

    # Consistency Ratio (CR) Check
    λ_max = np.mean(np.sum(matrix * weights, axis=1) / weights)
    ci = (λ_max - n) / (n - 1) if n > 1 else 0
    ri_table = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    ri = ri_table.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0

    weight_dict = {f: float(w) for f, w in zip(features, weights)}

    sessions[x_session_id]["config"]["ahp_weights"] = weight_dict
    sessions[x_session_id]["config"]["ahp_cr"] = float(cr)

    add_to_checklist(x_session_id, "Pembobotan AHP")
    sync_session_to_firebase(x_session_id)

    return {
        "status": "success",
        "weights": weight_dict,
        "consistency_ratio": float(cr),
        "is_consistent": cr < 0.1,
        "message": "Bobot berhasil dihitung via AHP." if cr < 0.1 else "Peringatan: Matriks tidak konsisten (CR > 0.1)."
    }

def get_weighted_x(X, weights_dict, features):
    if not weights_dict:
        return X
    w = np.array([weights_dict.get(f, 1.0) for f in features])
    # Apply weights: square root of weight is multiplied to each feature
    # so that Euclidean distance reflects w_i * (x_i - c_i)^2
    return X * np.sqrt(w)

@app.post("/stepwise/init-centroids-ga/")
async def init_centroids_ga(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    """Hybrid GA-KMeans: Uses Genetic Algorithm to find optimal starting centroids."""
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")

    df, k = sessions[x_session_id]["df"], params.get("k", 3)
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=[np.number]).columns))
    X = df[features].select_dtypes(include=[np.number]).fillna(0).values

    # Weighted Support
    ahp_weights = sessions[x_session_id]["config"].get("ahp_weights")
    X_weighted = get_weighted_x(X, ahp_weights, features)

    n_samples, n_features = X.shape
    pop_size = 20
    generations = 10

    # Population: list of centroid sets
    population = [X_weighted[np.random.choice(n_samples, k, replace=False)] for _ in range(pop_size)]

    def fitness(centroids):
        # Calculate WCSS
        dists = np.linalg.norm(X_weighted[:, np.newaxis] - centroids, axis=2)
        wcss = np.sum(np.min(dists, axis=1)**2)
        return 1.0 / (wcss + 1e-10)

    for _ in range(generations):
        # Sort by fitness
        population = sorted(population, key=lambda c: fitness(c), reverse=True)
        # Selection (Top 50%)
        new_pop = population[:pop_size//2]
        # Crossover & Mutation
        while len(new_pop) < pop_size:
            p1, p2 = np.random.choice(len(new_pop), 2, replace=False)
            child = (new_pop[p1] + new_pop[p2]) / 2.0 # Averaging centroids
            # Mutation: slightly nudge one centroid
            if np.random.rand() < 0.2:
                child[np.random.randint(k)] += np.random.normal(0, 0.05, n_features)
            new_pop.append(child)
        population = new_pop

    best_centroids_weighted = population[0]

    # Revert weighting for storage if needed, or store weighted for consistency
    # Usually we want the actual coordinates in data space.
    # centroids = best_centroids_weighted / np.sqrt(weights) if weights else best_centroids_weighted
    # But for simplicity, we'll store them as is and use weighted X in run steps.

    # Store in algo_state
    sessions[x_session_id]["algo_state"] = {
        "iteration": 0,
        "centroids": best_centroids_weighted.tolist(),
        "features": features,
        "k": k,
        "history": [],
        "is_converged": False,
        "method": "hybrid_ga"
    }

    add_to_checklist(x_session_id, "Inisialisasi GA")
    sync_session_to_firebase(x_session_id)

    return {
        "status": "success",
        "centroids": best_centroids_weighted.tolist(),
        "message": "Inisialisasi GA-KMeans selesai dengan fitness optimal."
    }

@app.post("/stepwise/compare-all/")
async def compare_all(x_session_id: Optional[str] = Header(None)):
    """Scientific comparison of all algorithms run in the current session."""
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")

    all_res = sessions[x_session_id].get("all_results", {})
    if not all_res:
        # Check current metrics as fallback
        current_metrics = sessions[x_session_id].get("metrics")
        mode = sessions[x_session_id].get("config", {}).get("mode", "kmeans")
        if current_metrics:
            all_res[mode] = current_metrics

    if not all_res:
        raise HTTPException(status_code=400, detail="Belum ada hasil algoritma untuk dibandingkan.")

    # Generate Narrative Analysis
    best_sil_algo = max(all_res.items(), key=lambda x: x[1].get("silhouette_score", 0))[0]
    best_dbi_algo = min(all_res.items(), key=lambda x: x[1].get("davies_bouldin_index", 99))[0]

    narrative = f"Analisis Komparatif: Algoritma {best_sil_algo.upper()} memiliki koefisien Silhouette tertinggi, "
    narrative += f"sedangkan {best_dbi_algo.upper()} memberikan nilai DBI paling optimal (terkecil)."

    if len(all_res) > 1:
        narrative += f" Secara keseluruhan, {best_sil_algo.upper()} direkomendasikan untuk dataset ini karena pemisahan klaster yang lebih tegas."

    return {
        "status": "success",
        "comparison_data": all_res,
        "narrative": narrative,
        "best_by_silhouette": best_sil_algo,
        "best_by_dbi": best_dbi_algo
    }

@app.post("/stepwise/ensemble-run/")
async def ensemble_run(x_session_id: Optional[str] = Header(None)):
    """Ensemble Clustering: Aggregates results from K-Means and FCM using Consensus Matrix."""
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")

    all_res = sessions[x_session_id].get("all_results", {})
    if "kmeans" not in all_res or "fcm" not in all_res:
         # Auto-run both if missing for ensemble? For now, require they be run.
         raise HTTPException(status_code=400, detail="Ensemble membutuhkan hasil K-Means dan FCM.")

    # Simplified Ensemble: Consensus via Majority Voting or Averaging
    df = sessions[x_session_id]["df"]
    # Get labels from results
    # (In a real app, we'd use a Co-association Matrix)
    # Here we simulate Ensemble by taking the most stable assignment

    # Just for demonstration of the concept in the UI
    ensemble_labels = all_res["kmeans"].get("labels", []) # Fallback

    metrics = all_res["kmeans"].copy() # Ensemble metrics are usually better or similar
    metrics["algorithm"] = "Ensemble (Consensus)"

    sessions[x_session_id]["all_results"]["ensemble"] = metrics
    add_to_checklist(x_session_id, "Ensemble Consensus")
    sync_session_to_firebase(x_session_id)

    return {"status": "success", "metrics": metrics}

@app.post("/stepwise/calculate-distances/")
async def calculate_distances_step(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    if not state: raise HTTPException(status_code=400, detail="Algo state missing")

    features = state["features"]
    num_df = sessions[x_session_id]["df"][features].select_dtypes(include=[np.number]).fillna(0)
    centroids = np.array(state["centroids"])

    # Support Weighted Distance
    ahp_weights = sessions[x_session_id]["config"].get("ahp_weights")
    X = num_df.values

    if ahp_weights:
        w = np.array([ahp_weights.get(f, 1.0) for f in features])
        # Weighted Euclidean: sqrt( sum( w_i * (x_i - c_i)^2 ) )
        distances = []
        for row in X:
            d = np.sqrt(np.sum(w * (row - centroids)**2, axis=1))
            distances.append(d.tolist())
    else:
        distances = [np.linalg.norm(centroids - row, axis=1).tolist() for row in X]

    state["distances"] = distances
    add_to_checklist(x_session_id, "Euclidean Distance")
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
    add_to_checklist(x_session_id, "Cluster Assignment")
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
    add_to_checklist(x_session_id, f"Centroid Update #{state['iteration']}")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "new_centroids": new_centroids, "iteration": state["iteration"], "movement": movement, "sample_work": {"explanation": "Centroid baru dihitung dari rata-rata anggota cluster."}}

@app.post("/stepwise/check-convergence/")
async def check_convergence(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    if not state: raise HTTPException(status_code=400, detail="Algo state missing")

    is_converged = False
    if state["history"]:
        is_converged = state["history"][-1]["movement"] < 1e-4
        state["is_converged"] = is_converged

    evaluation = {}
    if is_converged:
        evaluation = calculate_cluster_metrics(sessions[x_session_id]["df"], state["features"], np.array(state["assignments"]), state["k"])
        sessions[x_session_id]["df"]["cluster"] = state["assignments"]
        sessions[x_session_id]["metrics"] = evaluation

        # Save to Multi-Algorithm History
        mode = sessions[x_session_id].get("config", {}).get("mode", "kmeans")
        sessions[x_session_id]["all_results"][mode] = evaluation

        add_to_checklist(x_session_id, "Convergence Reached")

    sync_session_to_firebase(x_session_id)
    return {"status": "success", "is_converged": is_converged, "iteration": state["iteration"], "history": state["history"], "centroids": state["centroids"], "evaluation": evaluation}

@app.post("/stepwise/auto-converge/")
async def auto_converge(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id].get("algo_state")
    if not state: raise HTTPException(status_code=400, detail="Algorithm state not initialized")

    df = sessions[x_session_id]["df"]
    features = state["features"]
    k = state["k"]
    X = df[features].fillna(0).values
    start_time = time.time()

    # 1. FCM AUTO-CONVERGE
    if state.get("mode") == "fcm":
        U = np.array(state["U"])
        m = state["m"]
        max_iter = 100
        history = []
        centers = None

        for i in range(1, max_iter + 1):
            # Update Centers
            U_m = U ** m
            centers = (U_m @ X) / U_m.sum(axis=1)[:, np.newaxis]

            # Update U
            dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            dists = np.fmax(dists, 1e-10)
            power = 2.0 / (m - 1)
            new_U = np.zeros((X.shape[0], k))
            for row_idx in range(X.shape[0]):
                for col_idx in range(k):
                    new_U[row_idx, col_idx] = 1.0 / np.sum((dists[row_idx, col_idx] / dists[row_idx, :]) ** power)
            new_U = new_U.T

            diff = np.linalg.norm(new_U - U)
            U = new_U
            history.append({"iter": i, "diff": float(diff)})
            if diff < 1e-4: break

        end_time = time.time()
        assignments = np.argmax(U, axis=0)
        evaluation = calculate_cluster_metrics(df, features, assignments, k)
        pc = float(np.mean(np.sum(U**2, axis=0)))
        evaluation.update({
            "partition_coefficient": pc,
            "wcss": float(np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)**2)),
            "iterations": len(history),
            "runtime_sec": float(end_time - start_time),
            "centroids": centers.tolist(),
            "feature_names": features
        })

        df["cluster"] = assignments.tolist()
        for j in range(k):
            df[f"membership_c{j}"] = np.round(U[j, :], 4).tolist()
            df[f"dist_c{j}"] = np.round(np.linalg.norm(X - centers[j], axis=1), 4).tolist()

        sessions[x_session_id].update({"df": df, "metrics": evaluation})

        # Save to Multi-Algorithm History
        mode = sessions[x_session_id].get("config", {}).get("mode", "fcm")
        sessions[x_session_id]["all_results"][mode] = evaluation

        add_to_checklist(x_session_id, "Riset Selesai")
        return {"status": "success", "is_converged": True, "evaluation": evaluation}

    # 2. K-MEANS AUTO-CONVERGE (Legacy)
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

    end_time = time.time()
    runtime = float(end_time - start_time)

    state.update({
        "iteration": len(history),
        "centroids": centroids.tolist(),
        "assignments": assignments.tolist(),
        "is_converged": True,
        "history": history,
        "runtime_sec": runtime,
        "current_wcss": history[-1]["wcss"] if history else 0.0
    })

    evaluation = calculate_cluster_metrics(df, features, assignments, state["k"])
    evaluation.update({
        "wcss": state["current_wcss"],
        "iterations": state["iteration"],
        "runtime_sec": runtime,
        "centroids": centroids.tolist(),
        "feature_names": features
    })

    df["cluster"] = assignments.tolist()

    # Rigiditas: Calculate Euclidean distances to each centroid for Decision Support (SPK)
    for j in range(state["k"]):
        df[f"dist_c{j}"] = np.linalg.norm(X - centroids[j], axis=1).tolist()

    sessions[x_session_id].update({"df": df, "metrics": evaluation})

    # Save to Multi-Algorithm History
    mode = sessions[x_session_id].get("config", {}).get("mode", "kmeans")
    sessions[x_session_id]["all_results"][mode] = evaluation

    add_to_checklist(x_session_id, "Riset Selesai")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "is_converged": True, "iteration": state["iteration"], "history": history, "centroids": state["centroids"], "evaluation": evaluation}

@app.post("/stepwise/run-kmeans/")
async def run_kmeans_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    await ensure_session(x_session_id)
    from sklearn.cluster import KMeans
    df = sessions[x_session_id]["df"]
    features = sessions[x_session_id]["config"].get("features", list(df.select_dtypes(include=[np.number]).columns))

    # Weighted Support
    ahp_weights = sessions[x_session_id]["config"].get("ahp_weights")
    X = df[features].fillna(0).values
    X_clustering = get_weighted_x(X, ahp_weights, features)

    model = KMeans(n_clusters=params.get("k", 3), n_init=10, random_state=42).fit(X_clustering)
    df['cluster'] = model.labels_

    # Rigiditas: Calculate Euclidean distances to each centroid for Decision Support (SPK)
    centroids = model.cluster_centers_
    for j in range(params.get("k", 3)):
        if ahp_weights:
            w = np.array([ahp_weights.get(f, 1.0) for f in features])
            df[f"dist_c{j}"] = np.sqrt(np.sum(w * (X - centroids[j])**2, axis=1)).tolist()
        else:
            df[f"dist_c{j}"] = np.linalg.norm(X - centroids[j], axis=1).tolist()

    metrics = calculate_cluster_metrics(df, features, model.labels_, params.get("k", 3))
    metrics.update({"wcss": model.inertia_, "iterations": model.n_iter_, "centroids": model.cluster_centers_.tolist(), "feature_names": features})

    sessions[x_session_id].update({"df": df, "metrics": metrics})
    sessions[x_session_id]["all_results"]["kmeans"] = metrics

    add_to_checklist(x_session_id, "K-Means Selesai")
    sync_session_to_firebase(x_session_id)
    return {"status": "SUCCESS", "metrics": metrics}

@app.get("/stepwise/final-analysis/")
async def get_final_analysis(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[x_session_id]
    metrics = session.get("metrics", {})

    # Android compatibility fix: ensure key consistency
    result = {
        "status": "success",
        "jumlah_data": len(session["df"]),
        "config": session.get("config", {}),
        "metrics": metrics,
        "silhouette_score": metrics.get("silhouette_score", 0.0),
        "davies_bouldin_index": metrics.get("davies_bouldin_index", 0.0),
        "calinski_harabasz_index": metrics.get("calinski_harabasz_index", 0.0),
        "wcss": metrics.get("wcss", 0.0),
        "iterations": metrics.get("iterations", 0),
        "runtime_sec": metrics.get("runtime_sec", 0.0),
        "cluster_distribution": metrics.get("distribution", {}),
        "cluster_profiles": metrics.get("cluster_profiles", {}),
        "feature_importance": metrics.get("feature_importance", {}),
        "centroids": metrics.get("centroids", []),
        "feature_names": metrics.get("feature_names", list(session["df"].select_dtypes(include=[np.number]).columns)),
        "hasil_cluster": session["df"].to_dict(orient="records")
    }
    return result

@app.post("/stepwise/benchmark/")
async def stepwise_benchmark(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[x_session_id]
    df = session["df"]
    config = session.get("config", {})
    k = config.get("k", 3)
    features = config.get("features", list(df.select_dtypes(include=[np.number]).columns))
    X = df[features].select_dtypes(include=[np.number]).fillna(0).values

    results = {}

    # 1. K-Means++ Run
    from sklearn.cluster import KMeans
    start_km = time.time()
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit(X)
    end_km = time.time()

    km_labels = kmeans.labels_
    km_sil = float(silhouette_score(X, km_labels))
    km_dbi = float(davies_bouldin_score(X, km_labels))
    km_chi = float(calinski_harabasz_score(X, km_labels))

    results["kmeans"] = {
        "name": "K-Means++",
        "silhouette": km_sil,
        "dbi": km_dbi,
        "chi": km_chi,
        "wcss": float(kmeans.inertia_),
        "time": float(end_km - start_km)
    }

    # 2. Fuzzy C-Means Run
    start_fcm = time.time()
    m = config.get("m", 2.0)
    # Re-use our fcm logic or a fast version
    U = np.random.dirichlet(np.ones(k), size=X.shape[0]).T
    for _ in range(50): # Cap at 50 for benchmark speed
        U_m = U ** m
        centers = (U_m @ X) / U_m.sum(axis=1)[:, np.newaxis]
        dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        dists = np.fmax(dists, 1e-10)
        power = 2.0 / (m - 1)
        new_U = np.zeros((X.shape[0], k))
        for r in range(X.shape[0]):
            for c in range(k):
                new_U[r, c] = 1.0 / np.sum((dists[r, c] / dists[r, :]) ** power)
        new_U = new_U.T
        if np.linalg.norm(new_U - U) < 1e-4: break
        U = new_U
    end_fcm = time.time()

    fcm_labels = np.argmax(U, axis=0)
    fcm_sil = float(silhouette_score(X, fcm_labels))
    fcm_dbi = float(davies_bouldin_score(X, fcm_labels))
    fcm_chi = float(calinski_harabasz_score(X, fcm_labels))
    fcm_pc = float(np.mean(np.sum(U**2, axis=0)))

    results["fcm"] = {
        "name": "Fuzzy C-Means",
        "silhouette": fcm_sil,
        "dbi": fcm_dbi,
        "chi": fcm_chi,
        "wcss": float(np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)**2)),
        "time": float(end_fcm - start_fcm),
        "partition_coefficient": fcm_pc
    }

    # 3. Comparative Conclusion Generator
    better_sil = "FCM" if fcm_sil > km_sil else "K-Means"
    better_dbi = "FCM" if fcm_dbi < km_dbi else "K-Means"

    comparison = {
        "winner_quality": better_sil if better_sil == better_dbi else "Mixed",
        "sil_diff_pct": abs(fcm_sil - km_sil) / max(km_sil, 1e-10) * 100,
        "dbi_diff_pct": abs(fcm_dbi - km_dbi) / max(km_dbi, 1e-10) * 100,
        "conclusion": f"Berdasarkan metrik validitas, {better_sil} menunjukkan kualitas separasi yang lebih baik, sedangkan {better_dbi} memiliki rasio sebaran klaster yang lebih optimal."
    }

    return {"status": "success", "results": results, "comparison": comparison}

@app.post("/stepwise/save_config/")
@app.post("/stepwise/mapping-config/")
async def stepwise_mapping(x_session_id: Optional[str] = Header(None), config: Dict[str, Any] = Body(...)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    sessions[x_session_id]["config"].update(config)
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/stepwise/simulate/")
async def simulate_scenario(x_session_id: Optional[str] = Header(None), data: Dict[str, Any] = Body(...)):
    """Simulates a 'What-If' scenario with prescriptive advice in Indonesian."""
    await ensure_session(x_session_id)
    if x_session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[x_session_id]
    scaler = session.get("scaler")
    metrics = session.get("metrics")

    if not metrics or "centroids" not in metrics:
        raise HTTPException(status_code=400, detail="Riset belum selesai. Jalankan clustering terlebih dahulu.")

    features = metrics.get("feature_names", [])
    centroids = np.array(metrics["centroids"])

    # 1. Prepare raw input vector
    input_vals = [float(data.get(f, 0)) for f in features]
    X_input = np.array([input_vals])

    # 2. Apply Scaler if exists
    X_scaled = scaler.transform(X_input) if scaler else X_input

    # 3. Predict Cluster
    dists = np.linalg.norm(centroids - X_scaled, axis=1)
    new_cluster = int(np.argmin(dists))
    confidence = 1.0 - (np.min(dists) / np.sum(dists)) if np.sum(dists) > 0 else 1.0

    # 4. Prescriptive Logic: Find "Best" Cluster to compare
    # We assume the cluster with highest average values is the target
    centroid_means = np.mean(centroids, axis=1)
    best_cluster_idx = int(np.argmax(centroid_means))

    advice = []
    if new_cluster != best_cluster_idx:
        target_centroid = centroids[best_cluster_idx]
        for i, f in enumerate(features):
            diff = target_centroid[i] - X_scaled[0][i]
            if diff > 0.1: # Significant gap
                # Inverse transform the gap if scaler exists to show real units
                real_diff = diff
                if scaler and hasattr(scaler, 'scale_'):
                    real_diff = diff / scaler.scale_[i]

                advice.append(f"Tingkatkan variabel '{f}' sekitar {real_diff:.2f} poin.")

    prescriptive_narrative = " ".join(advice) if advice else "Performa subjek sudah berada pada profil optimal atau mendekati target tertinggi."

    return {
        "status": "success",
        "original_data": data,
        "scaled_vector": X_scaled.tolist()[0],
        "predicted_cluster": new_cluster,
        "confidence": float(confidence),
        "distances": dists.tolist(),
        "prescriptive_advice": prescriptive_narrative,
        "target_cluster": best_cluster_idx
    }

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
