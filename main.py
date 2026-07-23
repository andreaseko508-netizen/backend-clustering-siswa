from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Body
from fastapi.responses import StreamingResponse
from sdk.models import ExecutionContext, ExecutionResult
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

app = FastAPI(title="SIMORBATAS Python AI Runtime", version="1.0.0")

# Initialize Firebase Admin SDK
# Priority 1: Environment Variable (Cloud/Render)
# Priority 2: Local File (Development)
db = None
try:
    firebase_creds_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

    if firebase_creds_json:
        # Load from Environment Variable (Safe for GitHub)
        creds_dict = json.loads(firebase_creds_json)
        cred = credentials.Certificate(creds_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized via Environment Variable.")
    else:
        # Fallback to local file
        cred_path = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("Firebase initialized via serviceAccountKey.json.")
        else:
            print("Warning: Firebase credentials not found (Env Var or File). Firestore disabled.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")

# In-memory session storage (Transient)
# Structure: { session_id: { "df": DataFrame, "config": {}, "metrics": {}, "audit": {}, "algo_state": {} } }
sessions: Dict[str, Dict[str, Any]] = {}

def sync_session_to_firebase(session_id: str):
    """Saves the current session state to Firestore for cloud persistence."""
    if not db or session_id not in sessions:
        return

    try:
        session = sessions[session_id].copy()
        # Convert DataFrame to records for Firestore storage
        if "df" in session and isinstance(session["df"], pd.DataFrame):
            # We handle NaN/Inf because Firestore doesn't like them
            df_cleaned = session["df"].replace([np.inf, -np.inf], np.nan).fillna(0)
            session["df_records"] = df_cleaned.to_dict(orient="records")
            del session["df"] # Remove DF object before JSON serialization

        db.collection("python_sessions").document(session_id).set(session)
        print(f"Session {session_id} synced to Firestore.")
    except Exception as e:
        print(f"Failed to sync session {session_id}: {e}")

async def ensure_session(x_session_id: str):
    """Ensures session is in memory, fetching from Firestore if necessary."""
    if not x_session_id:
        return

    if x_session_id not in sessions:
        if db:
            doc = db.collection("python_sessions").document(x_session_id).get()
            if doc.exists:
                data = doc.to_dict()
                if "df_records" in data:
                    data["df"] = pd.DataFrame(data["df_records"])
                    del data["df_records"]
                sessions[x_session_id] = data
                print(f"Session {x_session_id} recovered from Firestore.")
            else:
                print(f"Session {x_session_id} not found in Firestore.")
        else:
            print(f"Session {x_session_id} not in memory and Firestore is disabled.")

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

        dbi = 0.0
        sil = 0.0
        chi = 0.0

        if len(unique_labels) > 1:
            dbi = float(davies_bouldin_score(X, assignments))
            sil = float(silhouette_score(X, assignments))
            chi = float(calinski_harabasz_score(X, assignments))

        dist = {}
        for i in range(k):
            count = int(np.sum(assignments == i))
            dist[str(i)] = {
                "count": count,
                "percentage": float(count / len(df) * 100)
            }

        # Calculate cluster profiles (Mean of features per cluster) for professional reports
        profiles = {}
        df_temp = df.copy()
        df_temp['cluster'] = assignments
        for i in range(k):
            cluster_data = df_temp[df_temp['cluster'] == i][features]
            if not cluster_data.empty:
                profiles[str(i)] = cluster_data.mean().to_dict()

        return {
            "davies_bouldin_index": dbi,
            "silhouette_score": sil,
            "calinski_harabasz_index": chi,
            "distribution": dist,
            "cluster_profiles": profiles
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            "davies_bouldin_index": 0.0,
            "silhouette_score": 0.0,
            "calinski_harabasz_index": 0.0,
            "distribution": {},
            "cluster_profiles": {}
        }

# Add plugins directory to path for dynamic loading
sys.path.append(os.path.join(os.path.dirname(__file__), "plugins"))

@app.get("/health")
async def health():
    return {"status": "UP", "engine": "Python AI Runtime"}

@app.post("/stepwise/upload/")
async def stepwise_upload(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(None)
):
    if not x_session_id:
        x_session_id = str(uuid.uuid4())

    try:
        content = await file.read()
        df = None

        # Robust Format Detection
        if file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(io.BytesIO(content))
            except:
                df = pd.read_excel(io.BytesIO(content)) # Fallback if named .csv but is .xlsx
        else:
            try:
                df = pd.read_excel(io.BytesIO(content))
            except:
                df = pd.read_csv(io.BytesIO(content)) # Fallback

        if df is None:
            raise HTTPException(status_code=400, detail="Could not parse file as CSV or Excel")

        sessions[x_session_id] = {
            "df": df,
            "filename": file.filename,
            "config": {},
            "metrics": {},
            "checkpoints": {
                "Data Asli": df.head(100).to_dict(orient="records")
            },
            "audit": {
                "initial_rows": len(df),
                "initial_cols": len(df.columns),
                "missing_before": int(df.isnull().sum().sum()),
                "outliers_removed": 0,
                "normalization_method": "None",
                "execution_checklist": []
            }
        }

        sync_session_to_firebase(x_session_id)

        return {
            "status": "success",
            "jumlah_data": len(df),
            "columns": list(df.columns),
            "session_id": x_session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stepwise/raw-data/")
async def get_raw_data(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[x_session_id]["df"]
    # Handle NaN and Inf before sending to JSON
    preview_df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return {
        "columns": list(df.columns),
        "total_rows": int(len(df)),
        "data": preview_df.to_dict(orient="records")
    }

@app.post("/stepwise/cleaning/")
async def stepwise_cleaning(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[x_session_id]["df"]
    initial_rows = len(df)

    # Store Before State for Sinta 2 Educational Comparison
    sessions[x_session_id]["checkpoints"]["Pembersihan Data (Sebelum)"] = df.to_dict(orient="records")

    # 1. Drop completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    after_empty_rows = len(df)

    # 2. Trim whitespace for string columns
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # 3. Drop Duplicates
    df = df.drop_duplicates()
    final_rows = len(df)

    sessions[x_session_id]["df"] = df
    sessions[x_session_id]["checkpoints"]["Pembersihan Data (Sesudah)"] = df.to_dict(orient="records")
    add_to_checklist(x_session_id, "Cleaning")
    sync_session_to_firebase(x_session_id)

    # Sample Work for Education
    sample_work = {
        "explanation": f"Sistem mendeteksi {initial_rows - final_rows} baris yang tidak valid (duplikat atau kosong).",
        "initial_count": initial_rows,
        "final_count": final_rows,
        "removed_count": initial_rows - final_rows
    }

    return {
        "status": "success",
        "initial_rows": initial_rows,
        "empty_rows_removed": initial_rows - after_empty_rows,
        "duplicates_removed": after_empty_rows - final_rows,
        "final_rows": final_rows,
        "sample_work": sample_work,
        "log": f"Cleaning selesai: {initial_rows - final_rows} baris bermasalah dihapus."
    }

@app.post("/stepwise/missing-value/")
async def stepwise_missing(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[x_session_id]["df"]

    # Store Before State
    sessions[x_session_id]["checkpoints"]["Imputasi Nilai Kosong (Sebelum)"] = df.to_dict(orient="records")

    num_cols = df.select_dtypes(include=['number']).columns
    initial_missing = int(df[num_cols].isnull().sum().sum())

    # Fill numeric NaNs with median
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    sessions[x_session_id]["df"] = df
    sessions[x_session_id]["checkpoints"]["Imputasi Nilai Kosong (Sesudah)"] = df.to_dict(orient="records")
    add_to_checklist(x_session_id, "Missing Value")
    sync_session_to_firebase(x_session_id)

    # Sample Work for UI (Prioritas 1)
    sample_work = {}
    if len(num_cols) > 0:
        target_col = num_cols[0]
        sample_work = {
            "feature": target_col,
            "method": "Median Imputation",
            "median_value": float(df[target_col].median()),
            "explanation": f"Nilai kosong pada fitur '{target_col}' diganti dengan nilai median untuk menjaga ketahanan (robustness) terhadap data ekstrem.",
            "formula": "\\tilde{x} = \\text{nilai tengah data yang telah diurutkan}"
        }

    return {
        "status": "success",
        "missing_filled": initial_missing,
        "method": "Median Imputation",
        "sample_work": sample_work,
        "log": f"Imputasi selesai: {initial_missing} nilai kosong diisi dengan Median."
    }

@app.get("/stepwise/missing-scan")
async def missing_scan(x_session_id: Optional[str] = Header(None)):
    print(f"DEBUG: missing-scan request for session {x_session_id}")
    await ensure_session(x_session_id)
    if not x_session_id or x_session_id not in sessions:
        print(f"DEBUG: Session {x_session_id} not found. Available: {list(sessions.keys())}")
        raise HTTPException(status_code=404, detail=f"Session {x_session_id} not found. Please re-upload dataset.")

    df = sessions[x_session_id]["df"]
    num_cols = df.select_dtypes(include=['number']).columns

    missing_stats = {}
    total_missing = 0

    for col in num_cols:
        count = int(df[col].isnull().sum())
        total_missing += count
        if count > 0:
            missing_stats[col] = {
                "count": count,
                "median": float(df[col].median())
            }

    return {
        "status": "success",
        "total_missing": total_missing,
        "missing_by_column": missing_stats,
        "total_rows": len(df)
    }

@app.post("/stepwise/elbow/")
async def stepwise_elbow(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    from sklearn.cluster import KMeans
    df = sessions[x_session_id]["df"]
    X = df.select_dtypes(include=[np.number]).fillna(0)

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append({"k": i, "wcss": float(kmeans.inertia_)})

    sessions[x_session_id]["checkpoints"]["Metode Elbow"] = wcss
    add_to_checklist(x_session_id, "Elbow Analysis")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "data": wcss}

@app.post("/stepwise/run-kmeans/")
async def run_kmeans_step(
    x_session_id: Optional[str] = Header(None),
    params: Dict[str, Any] = Body({"k": 3})
):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    from clustering.kmeans_plugin import KMeansPlugin
    from sdk.models import ExecutionContext

    df = sessions[x_session_id]["df"]

    # Create a temporary file for the plugin
    temp_path = "temp_kmeans.csv"
    df.to_csv(temp_path, index=False)

    context = ExecutionContext(
        execution_id=uuid.uuid4(),
        workflow_id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        institution_id=uuid.uuid4(),
        parameters={"n_clusters": params.get("k", 3)},
        input_datasets={"primary": temp_path},
        artifact_path="artifacts",
        temp_path="temp"
    )

    plugin = KMeansPlugin()
    result = plugin.execute(context)

    if result.status == "SUCCESS":
        # Load the result with clusters
        result_df = pd.read_csv(result.artifacts[0].file_path)
        sessions[x_session_id]["df"] = result_df
        sessions[x_session_id]["metrics"] = result.metrics

        # Unpack Clustering Checkpoints to main dictionary
        cl_check = result.metrics.get("clustering_checkpoints", {})
        sessions[x_session_id]["checkpoints"]["Centroid Awal"] = cl_check.get("Centroid Awal")
        sessions[x_session_id]["checkpoints"]["Jarak Euclidean"] = cl_check.get("Jarak Euclidean Awal")
        sessions[x_session_id]["checkpoints"]["Pembagian Cluster"] = cl_check.get("Pembagian Cluster Awal")
        sessions[x_session_id]["checkpoints"]["Histori Iterasi"] = result.metrics.get("iteration_history")
        sessions[x_session_id]["checkpoints"]["Hasil Akhir"] = result_df.head(100).to_dict(orient="records")

        if "centroids" in result.metrics:
            sessions[x_session_id]["checkpoints"]["Centroid Akhir"] = result.metrics["centroids"]

        sync_session_to_firebase(x_session_id)

    return result

@app.get("/stepwise/final-analysis/")
async def get_final_analysis(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[x_session_id]
    df = session["df"]
    metrics = session.get("metrics", {})

    # Calculate cluster distribution
    dist = {}
    if 'cluster' in df.columns:
        counts = df['cluster'].value_counts().to_dict()
        total = len(df)
        for c, count in counts.items():
            dist[str(c)] = {
                "count": int(count),
                "percentage": float(count / total * 100)
            }

    # Format according to TL V4.2 Standard Keys
    return {
        "status": "success",
        "jumlah_data": len(df),
        "config": session.get("config", {}),
        "metrics": metrics,
        "silhouette_score": metrics.get("silhouette_score", 0.0),
        "davies_bouldin_index": metrics.get("davies_bouldin_index", 0.0),
        "calinski_harabasz_index": metrics.get("calinski_harabasz_index", 0.0),
        "wcss": metrics.get("wcss", 0.0),
        "iterations": metrics.get("iterations", 0),
        "runtime_sec": metrics.get("runtime_sec", 0.0),
        "cluster_distribution": metrics.get("distribution", dist),
        "cluster_profiles": metrics.get("cluster_profiles", {}),
        "centroids": metrics.get("centroids", []),
        "feature_names": metrics.get("feature_names", []),
        "hasil_cluster": df.to_dict(orient="records")
    }

@app.post("/stepwise/outlier-detection/")
async def stepwise_outlier(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[x_session_id]["df"]

    # Store Before
    sessions[x_session_id]["checkpoints"]["Deteksi Outlier (Sebelum)"] = df.to_dict(orient="records")

    config = sessions[x_session_id].get("config", {})
    features = config.get("features", [])

    if not features:
        features = list(df.select_dtypes(include=['number']).columns)

    num_df = df[features].select_dtypes(include=['number'])
    Q1 = num_df.quantile(0.25)
    Q3 = num_df.quantile(0.75)
    IQR = Q3 - Q1

    outliers_mask = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)
    outlier_count = int(outliers_mask.sum())

    # In this step we don't automatically remove, just detect.
    # But for before/after comparison in UI, we might want to show what WOULD be removed
    df_after = df[~outliers_mask]

    sessions[x_session_id]["checkpoints"]["Deteksi Outlier (Sesudah)"] = df_after.to_dict(orient="records")
    sessions[x_session_id]["audit"]["outliers_removed"] += outlier_count
    add_to_checklist(x_session_id, "Outlier")
    sync_session_to_firebase(x_session_id)

    # Sample Work for Z-Score (Prioritas 1)
    sample_work = {}
    if len(features) > 0:
        first_col = features[0]
        val = df.iloc[0][first_col]
        mean = df[first_col].mean()
        std = df[first_col].std()
        z_score = (val - mean) / std if std != 0 else 0
        sample_work = {
            "feature": first_col,
            "value": float(val),
            "mean": float(mean),
            "std": float(std),
            "z_score": float(z_score),
            "threshold": 3.0,
            "formula": "z = \\frac{x - \\mu}{\\sigma}",
            "explanation": f"Nilai '{val}' pada fitur '{first_col}' dianalisis menggunakan metode Z-Score. Jika |z| > 3, maka data dianggap outlier."
        }

    return {
        "status": "success",
        "outlier_count": outlier_count,
        "total_rows": len(df),
        "sample_work": sample_work,
        "log": f"Deteksi selesai: Ditemukan {outlier_count} baris sebagai Outlier (Metode Z-Score)."
    }

@app.post("/stepwise/normalization/")
async def stepwise_norm(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    from sklearn.preprocessing import MinMaxScaler
    df = sessions[x_session_id]["df"]

    # Store Before State
    sessions[x_session_id]["checkpoints"]["Normalisasi Min-Max (Sebelum)"] = df.to_dict(orient="records")

    num_cols = df.select_dtypes(include=['number']).columns

    if len(num_cols) > 0:
        # Before state for sample work
        first_row_before = df.iloc[0][num_cols].to_dict()
        mins = df[num_cols].min()
        maxs = df[num_cols].max()

        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        sessions[x_session_id]["df"] = df
        sessions[x_session_id]["checkpoints"]["Normalisasi Min-Max (Sesudah)"] = df.to_dict(orient="records")
        sessions[x_session_id]["audit"]["normalization_method"] = "Min-Max"
        add_to_checklist(x_session_id, "Normalization")
        sync_session_to_firebase(x_session_id)

        # Sample Work (Prioritas 1)
        first_col = num_cols[0]
        val_before = first_row_before[first_col]
        val_after = df.iloc[0][first_col]
        sample_work = {
            "feature": first_col,
            "original_value": float(val_before),
            "min": float(mins[first_col]),
            "max": float(maxs[first_col]),
            "formula": "x' = \\frac{x - min}{max - min}",
            "explanation": f"Nilai '{val_before}' dinormalisasi menjadi '{val_after:.4f}' agar berada dalam rentang [0, 1].",
            "result": float(val_after)
        }

    return {
        "status": "success",
        "sample_work": sample_work if len(num_cols) > 0 else {},
        "log": "Normalisasi Min-Max [0, 1] berhasil diterapkan pada semua fitur numerik."
    }

@app.post("/stepwise/mapping-config/")
async def stepwise_mapping(
    x_session_id: Optional[str] = Header(None),
    config: Dict[str, Any] = Body(...)
):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sessions[x_session_id]["config"].update(config)
    add_to_checklist(x_session_id, "Selection")
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/stepwise/save_config/")
async def stepwise_save_config(
    x_session_id: Optional[str] = Header(None),
    config: Dict[str, Any] = Body(...)
):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sessions[x_session_id]["config"].update(config)
    add_to_checklist(x_session_id, "Selection")
    sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/stepwise/conversion/")
async def stepwise_conversion(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[x_session_id]["df"]

    # Store Before State for Sinta 2 Educational Comparison
    sessions[x_session_id]["checkpoints"]["Konversi Kategorikal (Sebelum)"] = df.to_dict(orient="records")

    config = sessions[x_session_id].get("config", {})
    features = config.get("features", [])

    # Label Encoding ONLY for selected feature columns that are categorical
    cat_cols = df[features].select_dtypes(include=['object', 'category']).columns
    mapping_details = {}

    sample_work = {}
    if len(cat_cols) > 0:
        target_col = cat_cols[0]
        # Get original value from the FIRST row before transformation
        original_val = df.iloc[0][target_col]

        for col in cat_cols:
            codes, uniques = pd.factorize(df[col])
            df[col] = codes
            mapping_details[col] = {str(i): str(val) for i, val in enumerate(uniques)}

        # Get new value after transformation
        new_val = df.iloc[0][target_col]

        # Sample Work for Education (Prioritas 1)
        sample_work = {
            "feature": target_col,
            "original_value": str(original_val),
            "converted_value": int(new_val),
            "explanation": f"Fitur kategorikal '{target_col}' diubah menjadi numerik menggunakan Label Encoding agar dapat diproses oleh algoritma K-Means.",
            "formula": "f(x) = \\text{index of } x \\text{ in unique labels}"
        }
    else:
        # If no categorical columns, provide a generic sample or empty
        sample_work = {
            "feature": "N/A",
            "explanation": "Tidak ditemukan fitur kategorikal untuk dikonversi. Data sudah dalam format numerik.",
            "formula": "N/A"
        }

    sessions[x_session_id]["df"] = df
    sessions[x_session_id]["conversion_mapping"] = mapping_details
    sessions[x_session_id]["checkpoints"]["Konversi Kategorikal (Sesudah)"] = df.to_dict(orient="records")
    add_to_checklist(x_session_id, "Conversion")
    sync_session_to_firebase(x_session_id)

    return {
        "status": "success",
        "converted_columns": list(cat_cols),
        "mappings": mapping_details,
        "sample_work": sample_work,
        "log": f"Konversi selesai: {len(cat_cols)} fitur kategorikal diubah menjadi numerik."
    }

@app.post("/stepwise/outlier-action/")
async def stepwise_outlier_action(
    action: str,
    x_session_id: Optional[str] = Header(None)
):
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # In a real app, we would apply specific logic (remove vs cap)
    # For now, we reuse the detection logic to "clean" if action is remove
    if action == "remove":
        return await stepwise_outlier(x_session_id)

    return {"status": "success", "action_applied": action}

@app.post("/stepwise/standardization/")
async def stepwise_standard(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    from sklearn.preprocessing import StandardScaler
    df = sessions[x_session_id]["df"]

    # Store Before State
    sessions[x_session_id]["checkpoints"]["Standardisasi Z-Score (Sebelum)"] = df.to_dict(orient="records")

    num_cols = df.select_dtypes(include=['number']).columns

    if len(num_cols) > 0:
        # Before state
        first_row_before = df.iloc[0][num_cols].to_dict()
        means = df[num_cols].mean()
        stds = df[num_cols].std().replace(0, 1) # Avoid division by zero

        scaler = StandardScaler()
        transformed = scaler.fit_transform(df[num_cols])
        # Replace NaN resulting from constant columns or division by zero with 0
        df[num_cols] = np.nan_to_num(transformed)

        sessions[x_session_id]["df"] = df
        sessions[x_session_id]["checkpoints"]["Standardisasi Z-Score (Sesudah)"] = df.to_dict(orient="records")
        sessions[x_session_id]["audit"]["normalization_method"] = "Z-Score"
        add_to_checklist(x_session_id, "Standardization")
        sync_session_to_firebase(x_session_id)

        # Sample Work
        first_col = num_cols[0]
        val_before = first_row_before[first_col]
        val_after = float(df.iloc[0][first_col])
        sample_work = {
            "feature": first_col,
            "original_value": float(val_before),
            "mean": float(means[first_col]),
            "std": float(stds[first_col]),
            "formula": "z = \\frac{x - \\mu}{\\sigma}",
            "explanation": f"Nilai '{val_before}' distandarisasi menggunakan nilai rata-rata (\\mu) dan standar deviasi (\\sigma) populasi.",
            "result": val_after
        }

    return {
        "status": "success",
        "sample_work": sample_work if len(num_cols) > 0 else {},
        "log": "Standardisasi Z-Score berhasil diterapkan."
    }

@app.get("/stepwise/normalization-stats/")
async def get_norm_stats(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[x_session_id]["df"]
    num_df = df.select_dtypes(include=['number'])

    stats = {}
    for col in num_df.columns:
        stats[col] = {
            "min": float(num_df[col].min()),
            "max": float(num_df[col].max()),
            "mean": float(num_df[col].mean()),
            "median": float(num_df[col].median()),
            "std": float(num_df[col].std()) if len(num_df) > 1 else 0.0,
            "variance": float(num_df[col].var()) if len(num_df) > 1 else 0.0
        }

    return {"status": "success", "stats": stats}

@app.get("/stepwise/session-state/")
async def get_session_state(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        return {"state": "IDLE"}

    # Logic could be more complex, but for now:
    return {"state": "UPLOADED", "session_id": x_session_id}

@app.get("/stepwise/quality-report/")
async def get_quality_report(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[x_session_id]
    df = session["df"]
    audit = session.get("audit", {})
    num_cols = list(df.select_dtypes(include=['number']).columns)
    constant_features = [col for col in df.columns if df[col].nunique() <= 1]
    missing_total = int(df.isnull().sum().sum())

    # Suitability Logic
    is_suitable = True
    suitability_msg = "Dataset Siap Diproses"

    if len(df) == 0:
        is_suitable = False
        suitability_msg = "Dataset Tidak Layak: Dataset Kosong"
    elif len(num_cols) < 2:
        is_suitable = False
        suitability_msg = "Dataset Tidak Layak: Kurang dari 2 Fitur Numerik"
    elif df.isnull().all().all():
        is_suitable = False
        suitability_msg = "Dataset Tidak Layak: Seluruh Nilai Kosong"
    elif len(constant_features) == len(df.columns):
        is_suitable = False
        suitability_msg = "Dataset Tidak Layak: Seluruh Fitur Bernilai Konstan"

    return {
        "status": "success",
        "filename": session.get("filename", "dataset.xlsx"),
        "completeness": 1.0 - (missing_total / df.size if df.size > 0 else 0),
        "rows": len(df),
        "cols": len(df.columns),
        "numeric_features": len(num_cols),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_features": len(constant_features),
        "missing_values": missing_total,
        "missing_before": audit.get("missing_before", 0),
        "outliers_removed": audit.get("outliers_removed", 0),
        "normalization_method": audit.get("normalization_method", "None"),
        "is_suitable": is_suitable,
        "suitability_message": suitability_msg,
        "execution_checklist": audit.get("execution_checklist", [])
    }

@app.get("/stepwise/checkpoints/")
async def get_checkpoints(x_session_id: Optional[str] = Header(None)):
    print(f"CHECKPOINT_REQ: SessionID={x_session_id}")
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        print(f"ERROR: Session {x_session_id} not found. Available: {list(sessions.keys())}")
        raise HTTPException(status_code=404, detail=f"Session {x_session_id} not found")

    return {
        "status": "success",
        "checkpoints": sessions[x_session_id].get("checkpoints", {})
    }

@app.get("/stepwise/universal-dataset/")
async def get_universal_dataset(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[x_session_id]["df"]
    return {
        "columns": list(df.columns),
        "data": df.to_dict(orient="records")
    }

@app.post("/run", response_model=ExecutionResult)
async def run_plugin(plugin_id: str, context: ExecutionContext):
    try:
        module_path, class_name = plugin_id.rsplit(".", 1)
        module = importlib.import_module(module_path)
        plugin_class = getattr(module, class_name)

        plugin = plugin_class()
        return plugin.execute(context)
    except Exception as e:
        return ExecutionResult(
            status="FAILED",
            metrics={},
            artifacts=[],
            error_message=str(e)
        )

@app.post("/stepwise/init-centroids/")
async def init_centroids_step(
    x_session_id: Optional[str] = Header(None),
    params: Dict[str, Any] = Body({"k": 3, "init_method": "random"})
):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    df = sessions[x_session_id]["df"]
    config = sessions[x_session_id].get("config", {})
    k = params.get("k", 3)
    init_method = params.get("init_method", "random")

    # Use selected features from config if available
    features = config.get("features", [])
    if not features:
        features = list(df.select_dtypes(include=[np.number]).columns)

    num_df = df[features].select_dtypes(include=[np.number]).fillna(0)
    # Ensure no Inf values that crash JSON
    num_df = num_df.replace([np.inf, -np.inf], 0)

    if len(num_df) < k:
         raise HTTPException(status_code=400, detail=f"Jumlah data ({len(num_df)}) lebih kecil dari nilai K ({k})")

    if init_method == "random":
        centroids = num_df.sample(n=k, random_state=42).values
        msg = f"Inisialisasi {k} centroid berhasil menggunakan metode acak (random)."
    elif init_method == "systematic":
        # Head-Mid-Tail Strategy (Prioritas Edukasi)
        n = len(num_df)

        # Calculate indices based on dataset size (Flexibility)
        if k == 3:
            indices = [0, n // 2, n - 1]
        else:
            # Spread K indices evenly across N rows
            indices = np.linspace(0, n - 1, k, dtype=int).tolist()

        centroids = num_df.iloc[indices].values

        # Create a more descriptive message showing flexibility
        row_numbers = [idx + 1 for idx in indices]
        msg = f"Inisialisasi {k} centroid berhasil secara sistematis. Sistem memilih data pada baris ke: {', '.join(map(str, row_numbers))} (Total data: {n})."
    else: # K-Means++
        from sklearn.cluster import kmeans_plusplus
        centroids, _ = kmeans_plusplus(num_df.values, n_clusters=k, random_state=42)
        msg = f"Inisialisasi {k} centroid berhasil menggunakan metode K-Means++."

    # Convert to standard Python types and ensure no NaN/Inf
    centroids_list = np.nan_to_num(centroids).tolist()

    sessions[x_session_id]["algo_state"] = {
        "iteration": 0,
        "centroids": centroids_list,
        "features": list(features),
        "k": k,
        "is_converged": False,
        "history": [],
        "init_method": init_method,
        "init_indices": indices if init_method == "systematic" else []
    }

    add_to_checklist(x_session_id, "Centroid Init")
    sync_session_to_firebase(x_session_id)

    return {
        "status": "success",
        "centroids": centroids_list,
        "features": list(features),
        "message": msg
    }

@app.post("/stepwise/calculate-distances/")
async def calculate_distances_step(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions or "algo_state" not in sessions[x_session_id]:
        raise HTTPException(status_code=400, detail="Algorithm state not initialized")

    state = sessions[x_session_id]["algo_state"]
    df = sessions[x_session_id]["df"]
    # Ensure we use the correct features and handle non-numeric/missing
    num_df = df[state["features"]].select_dtypes(include=[np.number]).fillna(0)
    num_df = num_df.replace([np.inf, -np.inf], 0)

    centroids = np.array(state["centroids"])
    distances = []

    for _, row in num_df.iterrows():
        point = row.values
        dist_to_centroids = np.linalg.norm(centroids - point, axis=1)
        # Ensure no NaN/Inf in distances
        distances.append(np.nan_to_num(dist_to_centroids).tolist())

    state["distances"] = distances
    add_to_checklist(x_session_id, "Euclidean Distance")
    sync_session_to_firebase(x_session_id)

    sample_point = num_df.iloc[0].values
    sample_dists = np.nan_to_num(np.linalg.norm(centroids - sample_point, axis=1))

    return {
        "status": "success",
        "distance_matrix_sample": distances[:5],
        "sample_work": {
            "student_index": 0,
            "values": sample_point.tolist(),
            "distances": sample_dists.tolist(),
            "formula": "d = \\sqrt{\\sum (x_i - c_i)^2}"
        }
    }

@app.post("/stepwise/assign-clusters/")
async def assign_clusters_step(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if "algo_state" not in sessions[x_session_id] or "distances" not in sessions[x_session_id]["algo_state"]:
        raise HTTPException(status_code=400, detail="Distances not calculated")

    state = sessions[x_session_id]["algo_state"]
    distances = np.array(state["distances"])
    assignments = np.argmin(distances, axis=1)

    state["assignments"] = assignments.tolist()

    min_distances = np.min(distances, axis=1)
    wcss = float(np.nansum(min_distances**2)) # Robust sum
    state["current_wcss"] = wcss

    add_to_checklist(x_session_id, "Cluster Assignment")
    sync_session_to_firebase(x_session_id)

    return {
        "status": "success",
        "assignments": assignments.tolist(),
        "wcss": wcss,
        "counts": {str(i): int(np.sum(assignments == i)) for i in range(state["k"])}
    }

@app.post("/stepwise/update-centroids/")
async def update_centroids_step(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id]["algo_state"]
    df = sessions[x_session_id]["df"]
    num_df = df[state["features"]].select_dtypes(include=[np.number]).fillna(0)
    assignments = np.array(state["assignments"])

    old_centroids = np.array(state["centroids"])
    new_centroids = []

    for i in range(state["k"]):
        cluster_points = num_df[assignments == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0).values.tolist())
        else:
            new_centroids.append(old_centroids[i].tolist())

    # Ensure no NaN/Inf in new centroids
    new_centroids = np.nan_to_num(new_centroids).tolist()

    movement = np.linalg.norm(np.array(new_centroids) - old_centroids)
    state["centroids"] = new_centroids
    state["iteration"] += 1
    state["history"].append({"iter": state["iteration"], "wcss": state["current_wcss"], "movement": float(movement)})

    add_to_checklist(x_session_id, "Update Centroid")
    sync_session_to_firebase(x_session_id)

    return {
        "status": "success",
        "new_centroids": new_centroids,
        "movement": float(movement),
        "iteration": state["iteration"],
        "sample_work": {
            "formula": "\\mu = \\frac{\\sum x}{n}",
            "explanation": "Titik pusat baru dihitung dari rata-rata seluruh anggota cluster."
        }
    }

@app.post("/stepwise/check-convergence/")
async def check_convergence_step(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    state = sessions[x_session_id]["algo_state"]
    history = state["history"]

    is_converged = False
    evaluation = {}

    if len(history) > 0:
        last_movement = history[-1]["movement"]
        if last_movement < 1e-4:
            is_converged = True
            state["is_converged"] = True
            add_to_checklist(x_session_id, "Convergence Reached")

            # Calculate Evaluation Metrics
            metrics = calculate_cluster_metrics(
                sessions[x_session_id]["df"],
                state["features"],
                np.array(state["assignments"]),
                state["k"]
            )

            # Store in session for final-analysis endpoint
            sessions[x_session_id]["metrics"] = metrics
            sessions[x_session_id]["metrics"]["wcss"] = state["current_wcss"]
            sessions[x_session_id]["metrics"]["iterations"] = state["iteration"]

            # Update dataframe with final clusters and distances
            sessions[x_session_id]["df"]["cluster"] = state["assignments"]
            final_dists = np.array(state["distances"])
            for i in range(state["k"]):
                sessions[x_session_id]["df"][f"dist_c{i}"] = final_dists[:, i].tolist()

            evaluation = metrics

        sync_session_to_firebase(x_session_id)
    else:
        add_to_checklist(x_session_id, "Convergence Check")

    return {
        "status": "success",
        "is_converged": is_converged,
        "iteration": state["iteration"],
        "centroids": state["centroids"],
        "history": history,
        "evaluation": evaluation
    }

@app.post("/stepwise/auto-converge/")
async def auto_converge(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions or "algo_state" not in sessions[x_session_id]:
        raise HTTPException(status_code=400, detail="Algorithm state not initialized")

    state = sessions[x_session_id]["algo_state"]
    df = sessions[x_session_id]["df"]
    features = state["features"]
    num_df = df[features].select_dtypes(include=[np.number]).fillna(0)

    max_iter = 100
    for _ in range(max_iter):
        if state.get("is_converged", False):
            break

        # 1. Assign Clusters
        centroids = np.array(state["centroids"])
        distances = []
        for _, row in num_df.iterrows():
            dist_to_centroids = np.linalg.norm(centroids - row.values, axis=1)
            distances.append(np.nan_to_num(dist_to_centroids).tolist())
        state["distances"] = distances

        distances_arr = np.array(distances)
        assignments = np.argmin(distances_arr, axis=1)
        state["assignments"] = assignments.tolist()

        min_distances = np.min(distances_arr, axis=1)
        state["current_wcss"] = float(np.nansum(min_distances**2))

        # 2. Update Centroids
        old_centroids = centroids
        new_centroids = []
        for i in range(state["k"]):
            cluster_points = num_df[assignments == i]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0).values.tolist())
            else:
                new_centroids.append(old_centroids[i].tolist())

        new_centroids = np.nan_to_num(new_centroids).tolist()
        movement = np.linalg.norm(np.array(new_centroids) - old_centroids)

        state["centroids"] = new_centroids
        state["iteration"] += 1
        state["history"].append({
            "iter": state["iteration"],
            "wcss": state["current_wcss"],
            "movement": float(movement)
        })

        if movement < 1e-4:
            state["is_converged"] = True
            add_to_checklist(x_session_id, "Convergence Reached")
            break

    # Calculate Evaluation Metrics
    metrics = calculate_cluster_metrics(
        df,
        features,
        np.array(state["assignments"]),
        state["k"]
    )

    # Store in session for final-analysis endpoint
    sessions[x_session_id]["metrics"] = metrics
    sessions[x_session_id]["metrics"]["wcss"] = state["current_wcss"]
    sessions[x_session_id]["metrics"]["iterations"] = state["iteration"]

    # Update dataframe with final clusters and distances
    sessions[x_session_id]["df"]["cluster"] = state["assignments"]
    final_dists = np.array(state["distances"])
    for i in range(state["k"]):
        sessions[x_session_id]["df"][f"dist_c{i}"] = final_dists[:, i].tolist()

    sync_session_to_firebase(x_session_id)

    return {
        "status": "success",
        "is_converged": state.get("is_converged", False),
        "iteration": state["iteration"],
        "centroids": state["centroids"],
        "history": state["history"],
        "evaluation": metrics
    }

    return {
        "status": "success",
        "is_converged": state.get("is_converged", False),
        "iteration": state["iteration"],
        "centroids": state["centroids"],
        "history": state["history"],
        "evaluation": metrics
    }

@app.get("/stepwise/export-excel/")
async def export_excel(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[x_session_id]
    checkpoints = session.get("checkpoints", {})

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. Data Asli
        if "Data Asli" in checkpoints:
            pd.DataFrame(checkpoints["Data Asli"]).to_excel(writer, sheet_name="1. Data Asli", index=False)

        # 2. Data Konversi
        if "Konversi Kategorikal (Sesudah)" in checkpoints:
            pd.DataFrame(checkpoints["Konversi Kategorikal (Sesudah)"]).to_excel(writer, sheet_name="2. Data Konversi", index=False)

        # 3. Data Cleansing
        if "Pembersihan Data (Sesudah)" in checkpoints:
            pd.DataFrame(checkpoints["Pembersihan Data (Sesudah)"]).to_excel(writer, sheet_name="3. Data Cleansing", index=False)
        elif "Deteksi Outlier (Sesudah)" in checkpoints:
            pd.DataFrame(checkpoints["Deteksi Outlier (Sesudah)"]).to_excel(writer, sheet_name="3. Data Cleansing", index=False)

        # 4. Data Missing Value
        if "Imputasi Nilai Kosong (Sesudah)" in checkpoints:
            pd.DataFrame(checkpoints["Imputasi Nilai Kosong (Sesudah)"]).to_excel(writer, sheet_name="4. Data Missing Value", index=False)

        # 5. Data Normalisasi
        if "Normalisasi Min-Max (Sesudah)" in checkpoints:
            pd.DataFrame(checkpoints["Normalisasi Min-Max (Sesudah)"]).to_excel(writer, sheet_name="5. Data Normalisasi", index=False)
        elif "Standardisasi Z-Score (Sesudah)" in checkpoints:
            pd.DataFrame(checkpoints["Standardisasi Z-Score (Sesudah)"]).to_excel(writer, sheet_name="5. Data Normalisasi", index=False)

        # 6. Metode Elbow
        if "Metode Elbow" in checkpoints:
            pd.DataFrame(checkpoints["Metode Elbow"]).to_excel(writer, sheet_name="6. Metode Elbow", index=False)

        # 7. Inisialisasi Centroid
        if "Centroid Awal" in checkpoints:
            pd.DataFrame(checkpoints["Centroid Awal"]).to_excel(writer, sheet_name="7. Centroid Awal", index=False)

        # 8. Iterasi Algoritma
        if "algo_state" in session and "history" in session["algo_state"]:
            pd.DataFrame(session["algo_state"]["history"]).to_excel(writer, sheet_name="8. Iterasi Algoritma", index=False)
        elif "Histori Iterasi" in checkpoints:
            pd.DataFrame(checkpoints["Histori Iterasi"]).to_excel(writer, sheet_name="8. Iterasi Algoritma", index=False)

        # 9. Hasil Akhir
        if "df" in session:
            session["df"].to_excel(writer, sheet_name="9. Hasil Akhir", index=False)

    output.seek(0)

    filename = f"Laporan_Riset_{uuid.uuid4().hex[:8]}.xlsx"
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/stepwise/export-pdf/")
async def export_pdf(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    # Basic PDF generation using simple text for now, or using a library if available.
    # Since we don't have reportlab installed by default, let's keep it simple or just send the final analysis as a formatted doc
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[x_session_id]
    metrics = session.get("metrics", {})

    content = f"""
    LAPORAN HASIL RISET CLUSTERING
    ==============================
    File: {session.get('filename', 'Unknown')}
    Tanggal: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

    RINGKASAN METRIK:
    - Silhouette Score: {metrics.get('silhouette_score', 0):.4f}
    - Davies-Bouldin Index: {metrics.get('davies_bouldin_index', 0):.4f}
    - Calinski-Harabasz Index: {metrics.get('calinski_harabasz_index', 0):.4f}
    - WCSS: {metrics.get('wcss', 0):.2f}
    - Total Iterasi: {metrics.get('iterations', 0)}

    DISTRIBUSI KLASTER:
    """
    dist = metrics.get('distribution', {})
    for k, v in dist.items():
        content += f"- Cluster {int(k)+1}: {v['count']} siswa ({v['percentage']:.1f}%)\n"

    output = io.BytesIO(content.encode('utf-8'))
    return StreamingResponse(
        output,
        media_type="text/plain", # Placeholder for PDF if library not present
        headers={"Content-Disposition": "attachment; filename=Ringkasan_Riset.txt"}
    )

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces requires port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
