import os
import sys
import tempfile

# Serverless environment (Vercel / AWS Lambda) fix for Matplotlib
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib_config')
import matplotlib
matplotlib.use('Agg')

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Body, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
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

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMORBATAS")

# --- DYNAMIC PATH RECOVERY ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- PROFESSIONAL MODULE IMPORT ---
try:
    from utils import (
        sessions, audit_checkpoints, db, ensure_session, ensure_audit,
        sync_session_to_firebase, add_to_checklist, get_representative_data, safe_float
    )
    from statistics import (
        calculate_cluster_metrics, run_real_ga_init, get_weighted_x,
        perform_normality_test_expert, calculate_hopkins, calculate_ahp_weights_and_cr,
        calculate_xie_beni, calculate_partition_entropy, perform_stability_audit, perform_sensitivity_audit
    )
    from reports import ResearchReportPDF
except ImportError:
    from api.utils import sessions, audit_checkpoints, db, ensure_session, ensure_audit, sync_session_to_firebase, add_to_checklist, get_representative_data, safe_float
    from api.statistics import calculate_cluster_metrics, run_real_ga_init, get_weighted_x, perform_normality_test_expert, calculate_hopkins, calculate_ahp_weights_and_cr, calculate_xie_beni, calculate_partition_entropy, perform_stability_audit, perform_sensitivity_audit
    from api.reports import ResearchReportPDF

app = FastAPI(title="SIMORBATAS Professional AI Runtime", version="22.0.0", redirect_slashes=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_valid_session(x_session_id: str):
    await ensure_session(x_session_id)
    if x_session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesi riset terputus. Silakan unggah dataset kembali.")
    return sessions[x_session_id]

def reorder_clusters_by_quality(df, features, assignments, k):
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
@app.get("/health")
async def root():
    return {"status": "Active", "engine": "SIMORBATAS-Vercel", "version": "22.0.0"}

# --- 1. DATA ACQUISITION & MAPPING ---

@app.post("/stepwise/upload/")
@app.post("/stepwise/upload")
async def stepwise_upload(file: UploadFile = File(...), x_session_id: Optional[str] = Header(None)):
    if not x_session_id: x_session_id = str(uuid.uuid4())
    try:
        content = await file.read()
        if not content:
            raise HTTPException(400, "Berkas kosong. Silakan pilih berkas Excel (.xlsx) yang berisi data.")

        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        df = None

        if file_ext in ['xlsx', 'xls'] or file_ext == '':
            try:
                df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
            except Exception:
                try:
                    df = pd.read_excel(io.BytesIO(content))
                except Exception:
                    try:
                        df = pd.read_csv(io.BytesIO(content))
                    except Exception:
                        pass

        if df is None or not isinstance(df, pd.DataFrame):
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception:
                raise HTTPException(400, f"Gagal membaca file {file.filename}. Format tidak dapat diproses.")

        if df.empty:
            raise HTTPException(400, "Dataset kosong. Pastikan berkas memiliki minimal 1 baris data.")

        clean_cols = []
        for c in df.columns:
            c_str = str(c).strip() if c is not None else "Col"
            if c_str.lower() in ['nan', 'none', '']:
                c_str = "Unnamed"
            clean_cols.append(c_str)
        df.columns = clean_cols

        df = df.replace([np.inf, -np.inf], np.nan)

        sessions[x_session_id] = {
            "df": df, "filename": file.filename, "config": {"k": 3, "features": [], "ahp_weights": {}},
            "start_time": time.time(), "metrics": {}, "audit": {"execution_checklist": []},
            "algo_state": {"iteration": 0, "history": []}
        }
        ensure_audit(x_session_id)
        try:
            audit_checkpoints[x_session_id]["01_Data_Asli"] = df.copy()
        except Exception: pass

        try:
            sync_session_to_firebase(x_session_id)
        except Exception: pass

        return {"status": "success", "session_id": x_session_id, "jumlah_data": len(df), "columns": [str(c) for c in df.columns]}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Upload Error: {str(e)}")
        raise HTTPException(500, f"Gagal membaca dataset: {str(e)}")

@app.get("/stepwise/raw-data/")
@app.get("/stepwise/raw-data")
async def get_raw_data(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    return {"columns": list(session["df"].columns), "total_rows": len(session["df"]), "data": get_representative_data(session["df"])}

@app.post("/stepwise/mapping-config/")
@app.post("/stepwise/mapping-config")
async def stepwise_mapping(x_session_id: Optional[str] = Header(None), x_analysis_mode: Optional[str] = Header(None), config: Dict[str, Any] = Body(...)):
    session = await get_valid_session(x_session_id)
    session["config"].update(config)
    mode = x_analysis_mode or config.get("analysis_mode", session["config"].get("analysis_mode", "thesis"))
    session["config"]["analysis_mode"] = str(mode).lower()
    ensure_audit(x_session_id)
    if "features" in config: audit_checkpoints[x_session_id]["02_Seleksi_Variabel"] = session["df"][config["features"]].copy()
    add_to_checklist(x_session_id, "Seleksi Variabel"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "analysis_mode": session["config"]["analysis_mode"]}

@app.post("/stepwise/select-features/")
@app.post("/stepwise/select-features")
async def stepwise_select_features(x_session_id: Optional[str] = Header(None), columns: List[str] = Body(...)):
    session = await get_valid_session(x_session_id); session["config"]["features"] = columns; sync_session_to_firebase(x_session_id)
    return {"status": "success"}

@app.post("/stepwise/save_config/")
@app.post("/stepwise/save_config")
async def stepwise_save_config(x_session_id: Optional[str] = Header(None), x_analysis_mode: Optional[str] = Header(None), config: Dict[str, Any] = Body(...)):
    session = await get_valid_session(x_session_id)
    session["config"].update(config)
    mode = x_analysis_mode or config.get("analysis_mode", session["config"].get("analysis_mode", "thesis"))
    session["config"]["analysis_mode"] = str(mode).lower()
    add_to_checklist(x_session_id, "Konfigurasi Algoritma"); sync_session_to_firebase(x_session_id)
    return {"status": "success", "analysis_mode": session["config"]["analysis_mode"]}

@app.get("/stepwise/session-state/")
@app.get("/stepwise/session-state")
async def get_session_state(x_session_id: Optional[str] = Header(None)):
    await ensure_session(x_session_id)
    return {"state": "UPLOADED" if x_session_id in sessions else "IDLE"}

# --- 2. PREPROCESSING ---

@app.post("/stepwise/conversion/")
@app.post("/stepwise/conversion")
async def stepwise_conversion(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    df = session["df"]
    config = session.get("config", {})

    identity_col = config.get("identity", "")
    label_col = config.get("label", "")
    ignored_cols = config.get("ignored", [])
    features = config.get("features", [])

    # Metadata columns (NIS/ID, Nama, Ignored) are PROTECTED from mathematical conversion
    protected_cols = set([identity_col, label_col] + (ignored_cols if isinstance(ignored_cols, list) else []))
    protected_cols = {c for c in protected_cols if c}

    ORD_RULES = {
        "prestasi": {
            "tidak pernah": 0, "tidak ada": 0, "tidak": 0, "0": 0,
            "tingkat sekolah": 1, "sekolah": 1, "1": 1,
            "tingkat kecamatan": 2, "kecamatan": 2, "2": 2,
            "tingkat kabupaten": 3, "tingkat kabupaten/kota": 3, "kabupaten": 3, "kabupaten/kota": 3, "3": 3,
            "tingkat provinsi": 4, "provinsi": 4, "4": 4,
            "tingkat nasional": 5, "nasional": 5, "5": 5,
            "tingkat internasional": 6, "internasional": 6, "6": 6
        },
        "kendaraan": {
            "jalan kaki": 0, "jalan kaki/tidak ada": 0, "tidak ada": 0, "0": 0,
            "sepeda": 1, "1": 1,
            "motor": 2, "sepeda motor": 2, "2": 2,
            "mobil": 3, "3": 3,
            "angkutan umum": 4, "4": 4
        },
        "internet": {
            "tidak": 0, "tidak ada": 0, "tidak punya": 0, "ridak": 0, "0": 0,
            "ya": 1, "ada": 1, "punya": 1, "1": 1
        }
    }

    mappings = {}

    # Target ONLY selected feature columns or non-protected columns
    if features:
        target_cols = [c for c in features if c in df.columns and c not in protected_cols]
    else:
        target_cols = [c for c in df.columns if c not in protected_cols]

    for col in target_cols:
        # Robust check: column is non-numeric OR contains string/text values
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        has_strings = df[col].apply(lambda x: isinstance(x, str)).any()

        if not is_numeric or has_strings:
            col_lower = str(col).lower()
            matched_rule = None
            for rule_key, rule_map in ORD_RULES.items():
                if rule_key in col_lower:
                    matched_rule = rule_map
                    break

            if matched_rule:
                display_map = {}
                converted_vals = []

                for val in df[col]:
                    raw_str = str(val).strip() if pd.notnull(val) else ""
                    clean_str = raw_str.lower()

                    code_val = matched_rule.get(clean_str)
                    if code_val is None:
                        # Fallback substring match
                        code_val = 0
                        for k_rule, v_rule in matched_rule.items():
                            if k_rule in clean_str or clean_str in k_rule:
                                code_val = v_rule
                                break

                    display_map[raw_str if raw_str else "N/A"] = str(code_val)
                    converted_vals.append(code_val)

                mappings[col] = display_map
                df[col] = pd.Series(converted_vals, index=df.index).astype(int)
            else:
                codes, uniques = pd.factorize(df[col].astype(str))
                df[col] = pd.Series(codes, index=df.index).astype(int)
                mappings[col] = {str(u): str(i) for i, u in enumerate(uniques)}

    sample_work = {
        "explanation": "Transformasi kategorikal (Label Encoding & Ordinal Mapping) mengkonversi atribut fitur ke domain kuantitatif berurut. Kolom metadata (NIS/ID dan Nama Siswa) tidak dikonversi dan dikunci untuk identifikasi hasil akhir klaster.",
        "formula": r"f: \text{Kategori} \to \mathbb{Z}^+"
    }

    session["df"] = df
    add_to_checklist(x_session_id, "Konversi Fitur")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "mappings": mappings, "sample_work": sample_work}

@app.post("/stepwise/cleaning/")
@app.post("/stepwise/cleaning")
async def stepwise_cleaning(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); df = session["df"].dropna(how='all').drop_duplicates()
    session["df"] = df; add_to_checklist(x_session_id, "Pembersihan Data")
    return {"status": "success", "final_rows": len(df)}

@app.post("/stepwise/missing-value/")
@app.post("/stepwise/missing-value")
async def stepwise_missing(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    for col in session["df"].select_dtypes(include=['number']).columns:
        session["df"][col] = session["df"][col].fillna(session["df"][col].median())
    add_to_checklist(x_session_id, "Imputasi Data")
    return {"status": "success"}

@app.get("/stepwise/missing-scan/")
@app.get("/stepwise/missing-scan")
async def missing_scan(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); df = session["df"]
    stats = {col: {"count": int(df[col].isnull().sum()), "median": float(df[col].median())} for col in df.select_dtypes(include=['number']).columns if df[col].isnull().sum() > 0}
    return {"status": "success", "missing_by_column": stats}

@app.post("/stepwise/outlier-detection/")
@app.post("/stepwise/outlier-detection")
async def stepwise_outlier(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); num_df = session["df"].select_dtypes(include=['number'])
    if len(num_df) < 5: return {"status": "success", "outlier_count": 0}
    mask = (np.abs((num_df - num_df.mean()) / (num_df.std() + 1e-10)) > 3).any(axis=1)
    return {"status": "success", "outlier_count": int(mask.sum())}

@app.post("/stepwise/outlier-action/")
@app.post("/stepwise/outlier-action")
async def stepwise_outlier_action(x_session_id: Optional[str] = Header(None), action: str = Query("remove")):
    return {"status": "success", "action": action}

# --- 3. ANALYTICAL AUDIT ---

@app.get("/stepwise/quality-report/")
@app.get("/stepwise/quality-report")
async def get_quality_report(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); df = session["df"]
    num_cols = list(df.select_dtypes(include=['number']).columns)
    hopkins = calculate_hopkins(df[num_cols].fillna(0).values) if len(num_cols) >= 2 else 0.5
    completeness = 1.0 - (df.isnull().sum().sum() / df.size if df.size > 0 else 0)
    return {"status": "success", "rows": len(df), "cols": len(df.columns), "numeric_features": len(num_cols), "completeness": safe_float(completeness), "hopkins_statistic": safe_float(hopkins), "is_suitable": len(df) > 5, "execution_checklist": session["audit"].get("execution_checklist", [])}

@app.get("/stepwise/checkpoints/")
@app.get("/stepwise/checkpoints")
async def get_checkpoints(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    return {"status": "success", "checkpoints": session.get("checkpoints", {})}

@app.get("/stepwise/universal-dataset/")
@app.get("/stepwise/universal-dataset")
async def get_universal_dataset(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    df = session["df"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return {"status": "success", "columns": list(df.columns), "data": df.head(500).to_dict(orient="records")}

@app.get("/stepwise/normalization-stats/")
@app.get("/stepwise/normalization-stats")
async def get_norm_stats(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); num_df = session["df"].select_dtypes(include=['number'])
    stats = {col: {"min": safe_float(num_df[col].min()), "max": safe_float(num_df[col].max()), "mean": safe_float(num_df[col].mean()), "median": safe_float(num_df[col].median()), "std": safe_float(num_df[col].std()), "variance": safe_float(num_df[col].var())} for col in num_df.columns if not num_df[col].isnull().all()}
    return {"status": "success", "stats": stats}

@app.get("/stepwise/normality-test/")
@app.get("/stepwise/normality-test")
async def stepwise_normality_test(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); feats = session["config"].get("features", list(session["df"].columns))
    return perform_normality_test_expert(session["df"], feats)

@app.get("/stepwise/correlation-matrix/")
@app.get("/stepwise/correlation-matrix")
async def get_correlation_analysis(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); import matplotlib.pyplot as plt, seaborn as sns
    feats = session["config"].get("features", list(session["df"].columns)); num_df = session["df"][feats].select_dtypes(include=[np.number]).fillna(0)
    corr = num_df.corr().fillna(0); plt.figure(figsize=(10, 8)); sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close()
    high = [{"f1": corr.columns[i], "f2": corr.columns[j], "val": safe_float(corr.iloc[i, j])} for i in range(len(corr.columns)) for j in range(i) if abs(corr.iloc[i, j]) > 0.7]
    add_to_checklist(x_session_id, "Analisis Korelasi"); return {"status": "success", "heatmap_image": base64.b64encode(buf.getvalue()).decode('utf-8'), "high_correlations": high}

# --- 4. SCALING & OPTIMIZATION ---

@app.post("/stepwise/normalization/")
@app.post("/stepwise/normalization")
async def stepwise_norm(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    df = session["df"]

    # Store unscaled backup for clean re-scaling
    if "unscaled_df" not in session:
        session["unscaled_df"] = df.copy()
    else:
        df = session["unscaled_df"].copy()

    config = session.get("config", {})
    identity_col = config.get("identity", "")
    label_col = config.get("label", "")
    ignored_cols = config.get("ignored", [])
    features = config.get("features", [])

    protected_cols = set([identity_col, label_col] + (ignored_cols if isinstance(ignored_cols, list) else []))
    protected_cols = {c for c in protected_cols if c}

    if features:
        num_cols = [c for c in features if c in df.columns and c not in protected_cols]
    else:
        num_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in protected_cols]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    if len(num_cols) > 0:
        df[num_cols] = scaler.fit_transform(df[num_cols])
        session["scaler"] = scaler
        session["scaler_type"] = "minmax"

    session["df"] = df
    add_to_checklist(x_session_id, "Normalisasi Min-Max")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "scaler_type": "minmax", "columns_scaled": num_cols}


@app.post("/stepwise/standardization/")
@app.post("/stepwise/standardization")
async def stepwise_standard(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    df = session["df"]

    if "unscaled_df" not in session:
        session["unscaled_df"] = df.copy()
    else:
        df = session["unscaled_df"].copy()

    config = session.get("config", {})
    identity_col = config.get("identity", "")
    label_col = config.get("label", "")
    ignored_cols = config.get("ignored", [])
    features = config.get("features", [])

    protected_cols = set([identity_col, label_col] + (ignored_cols if isinstance(ignored_cols, list) else []))
    protected_cols = {c for c in protected_cols if c}

    if features:
        num_cols = [c for c in features if c in df.columns and c not in protected_cols]
    else:
        num_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in protected_cols]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    if len(num_cols) > 0:
        df[num_cols] = scaler.fit_transform(df[num_cols])
        session["scaler"] = scaler
        session["scaler_type"] = "zscore"

    session["df"] = df
    add_to_checklist(x_session_id, "Standardisasi Z-Score")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "scaler_type": "zscore", "columns_scaled": num_cols}


@app.post("/stepwise/robust-scaling/")
@app.post("/stepwise/robust-scaling")
async def stepwise_robust_scaling(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    df = session["df"]

    if "unscaled_df" not in session:
        session["unscaled_df"] = df.copy()
    else:
        df = session["unscaled_df"].copy()

    config = session.get("config", {})
    identity_col = config.get("identity", "")
    label_col = config.get("label", "")
    ignored_cols = config.get("ignored", [])
    features = config.get("features", [])

    protected_cols = set([identity_col, label_col] + (ignored_cols if isinstance(ignored_cols, list) else []))
    protected_cols = {c for c in protected_cols if c}

    if features:
        num_cols = [c for c in features if c in df.columns and c not in protected_cols]
    else:
        num_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in protected_cols]

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    if len(num_cols) > 0:
        df[num_cols] = scaler.fit_transform(df[num_cols])
        session["scaler"] = scaler
        session["scaler_type"] = "robust"

    session["df"] = df
    add_to_checklist(x_session_id, "Robust Scaling")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "scaler_type": "robust", "columns_scaled": num_cols}

@app.post("/stepwise/ahp-calculate/")
@app.post("/stepwise/ahp-calculate")
async def ahp_calculate(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body(...)):
    session = await get_valid_session(x_session_id); consensus = np.exp(np.mean(np.log(np.fmax(np.stack([np.array(m) for m in params.get("matrices", [params.get("matrix")]) if m]), 1e-10)), axis=0))
    w, cr = calculate_ahp_weights_and_cr(consensus); weight_dict = {f: safe_float(wi) for f, wi in zip(params.get("features"), w)}
    session["config"].update({"ahp_weights": weight_dict, "ahp_cr": cr}); sync_session_to_firebase(x_session_id)
    return {"status": "success", "weights": weight_dict, "cr": cr}

@app.post("/stepwise/elbow/")
@app.post("/stepwise/elbow")
async def stepwise_elbow(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    config = session.get("config", {})
    feats = config.get("features", list(session["df"].select_dtypes(include=['number']).columns))
    protected_cols = set([config.get("identity", ""), config.get("label", "")] + (config.get("ignored", []) if isinstance(config.get("ignored", []), list) else []))
    feats = [f for f in feats if f in session["df"].columns and f not in protected_cols]

    X = get_weighted_x(session["df"][feats].fillna(0).values, config.get("ahp_weights"), feats)
    wcss = [{"k": i, "wcss": safe_float(KMeans(n_clusters=i, n_init=10, random_state=42).fit(X).inertia_)} for i in range(1, 11)]
    return {"status": "success", "data": wcss}

@app.post("/stepwise/gap-statistic/")
@app.post("/stepwise/gap-statistic")
async def stepwise_gap_statistic(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); X = session["df"].select_dtypes(include=[np.number]).fillna(0).values
    if len(X) < 10: return {"status": "success", "recommended_k": 3, "gap_values": []}
    n_s, n_f = X.shape; ks = range(1, 7); gaps = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X); log_wcss = np.log(km.inertia_ + 1e-10)
        ref = [np.log(KMeans(n_clusters=k, n_init=5, random_state=i).fit(np.random.uniform(X.min(axis=0), X.max(axis=0), size=(n_s, n_f))).inertia_ + 1e-10) for i in range(5)]
        gaps.append({"k": k, "gap": safe_float(np.mean(ref) - log_wcss)})
    return {"status": "success", "gap_values": gaps, "recommended_k": int(ks[np.argmax([g["gap"] for g in gaps])])}

@app.get("/stepwise/compare_k/")
@app.get("/stepwise/compare_k")
async def stepwise_compare_k(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); feats = session["config"].get("features", list(session["df"].columns)); X = get_weighted_x(session["df"][feats].fillna(0).values, session["config"].get("ahp_weights"), feats)
    res = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X); res.append({"k": k, "silhouette": safe_float(silhouette_score(X, km.labels_)), "dbi": safe_float(davies_bouldin_score(X, km.labels_))})
    return {"status": "success", "results": res, "best_k_dbi": min(res, key=lambda x: x["dbi"])["k"], "best_k_silhouette": max(res, key=lambda x: x["silhouette"])["k"]}

# --- 5. CLUSTERING CORE ---

@app.post("/stepwise/init-centroids/")
@app.post("/stepwise/init-centroids")
async def init_centroids_step(x_session_id: Optional[str] = Header(None), x_analysis_mode: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3})):
    session = await get_valid_session(x_session_id)
    config = session.get("config", {})

    mode = x_analysis_mode or config.get("analysis_mode", "thesis")
    config["analysis_mode"] = str(mode).lower()

    feats = config.get("features", list(session["df"].select_dtypes(include=['number']).columns))
    protected_cols = set([config.get("identity", ""), config.get("label", "")] + (config.get("ignored", []) if isinstance(config.get("ignored", []), list) else []))
    feats = [f for f in feats if f in session["df"].columns and f not in protected_cols]

    k = params.get("k", 3)
    method = str(params.get("init_method", params.get("strategy", "systematic"))).lower()

    if config["analysis_mode"] == "baseline":
        method = "systematic"
    elif "ga" not in method and "hybrid" not in method and method != "systematic":
        method = "hybrid_ga"

    X = get_weighted_x(session["df"][feats].fillna(0).values, config.get("ahp_weights"), feats)

    if "ga" in method or "hybrid" in method:
        best_seeds = run_real_ga_init(X, k, population_size=30, generations=25)
        msg = f"Inisialisasi Centroid Hybrid GA (Evolusi Populasi) Berhasil. K={k} Centroid Ter-optimasi."
        init_type = "Hybrid GA"
    else:
        n_samples = len(X)
        row_scores = np.sum(X, axis=1)
        sorted_indices = np.argsort(row_scores)
        percentiles = np.linspace(10, 90, k)

        sampled_centroids = []
        for p in percentiles:
            idx = int(np.clip(p / 100.0 * (n_samples - 1), 0, n_samples - 1))
            sampled_centroids.append(X[sorted_indices[idx]])

        best_seeds = np.array(sampled_centroids)
        msg = f"Inisialisasi Centroid Deterministik (Head-Mid-Tail) Berhasil. K={k} Centroid Terpilih."
        init_type = "Systematic Head-Mid-Tail"

    centroids_list = best_seeds.tolist()

    session["algo_state"] = {
        "iteration": 0,
        "centroids": centroids_list,
        "init_centroids": centroids_list,
        "features": feats,
        "k": k,
        "history": [],
        "init_type": init_type
    }

    add_to_checklist(x_session_id, f"Inisialisasi Centroid ({init_type})")
    sync_session_to_firebase(x_session_id)
    return {
        "status": "success",
        "centroids": centroids_list,
        "features": feats,
        "message": msg,
        "init_type": init_type,
        "analysis_mode": config["analysis_mode"]
    }

@app.post("/stepwise/calculate-distances/")
@app.post("/stepwise/calculate-distances")
async def calculate_distances_step(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id)
    state = session.get("algo_state")
    if not state or "centroids" not in state:
        return {"status": "error", "message": "Inisialisasi centroid belum dilakukan!"}

    feats = state["features"]
    centroids = np.array(state["centroids"])

    config = session.get("config", {})
    X = get_weighted_x(session["df"][feats].fillna(0).values, config.get("ahp_weights"), feats)

    dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2).tolist()
    state["distances"] = dists

    sample_matrix = dists[:50]
    sample_work = {
        "distances": dists[0],
        "explanation": "Jarak Euclidean dihitung sebagai L2 Norm antara vektor sampel siswa dan setiap centroid klaster."
    }

    add_to_checklist(x_session_id, "Kalkulasi Jarak Euclidean")
    sync_session_to_firebase(x_session_id)
    return {
        "status": "success",
        "distance_matrix_sample": sample_matrix,
        "sample_work": sample_work,
        "total_rows": len(dists)
    }

@app.post("/stepwise/assign-clusters/")
@app.post("/stepwise/assign-clusters")
async def assign_clusters_step(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); state = session["algo_state"]; dists = np.array(state["distances"]); assignments = np.argmin(dists, axis=1).tolist()
    state["assignments"] = assignments; session["df"]["cluster"] = assignments
    for j in range(state["k"]): session["df"][f"dist_c{j}"] = dists[:, j]
    return {"status": "success", "assignments": assignments}

@app.post("/stepwise/update-centroids/")
@app.post("/stepwise/update-centroids")
async def update_centroids_step(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); state = session["algo_state"]; X = get_weighted_x(session["df"][state["features"]].fillna(0).values, session["config"].get("ahp_weights"), state["features"])
    assign = np.array(state["assignments"]); new_c = [X[assign == i].mean(axis=0).tolist() if len(X[assign == i]) > 0 else state["centroids"][i] for i in range(state["k"])]
    move = safe_float(np.linalg.norm(np.array(new_c) - np.array(state["centroids"]))); state["centroids"], state["iteration"] = new_c, state["iteration"] + 1
    state["history"].append({"iter": state["iteration"], "movement": move, "wcss": safe_float(np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - np.array(new_c), axis=2), axis=1)**2))})
    return {"status": "success", "new_centroids": new_c, "iteration": state["iteration"], "movement": move}

@app.post("/stepwise/check-convergence/")
@app.post("/stepwise/check-convergence")
async def check_convergence_step(x_session_id: Optional[str] = Header(None)):
    """
    Step 20 - Phase 1: Single Step Convergence Check.
    Executes ONLY 1 K-Means iteration to show initial movement delta.
    """
    session = await get_valid_session(x_session_id)
    state = session.get("algo_state", {})

    config = session.get("config", {})
    feats = state.get("features", config.get("features", list(session["df"].select_dtypes(include=['number']).columns)))
    config_ignored = config.get("ignored", [])
    protected_cols = set([config.get("identity", ""), config.get("label", "")] + (config_ignored if isinstance(config_ignored, list) else []))
    feats = [f for f in feats if f in session["df"].columns and f not in protected_cols]

    k = state.get("k", config.get("k", 3))
    ahp = config.get("ahp_weights")

    X = get_weighted_x(session["df"][feats].fillna(0).values, ahp, feats)

    init_c = state.get("init_centroids")
    if init_c and len(init_c) == k:
        centroids = np.array(init_c)
    else:
        centroids = np.array(state.get("centroids", X[:k]))
        if len(centroids) != k:
            centroids = X[:k]

    # Execute SINGLE iteration (#1) to allow user to see movement first
    dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    assignments = np.argmin(dists, axis=1)
    wcss = safe_float(np.sum(np.min(dists, axis=1)**2))

    new_c = np.array([X[assignments == j].mean(axis=0) if len(X[assignments == j]) > 0 else centroids[j] for j in range(k)])
    move = safe_float(np.linalg.norm(new_c - centroids))

    history = [{
        "iter": 1,
        "movement": move,
        "wcss": wcss,
        "status": "Bergerak" if move >= 1e-4 else "STABIL"
    }]

    session["algo_state"] = {
        **state,
        "centroids": new_c.tolist(),
        "assignments": assignments.tolist(),
        "is_converged": False,
        "history": history,
        "iteration": 1
    }

    add_to_checklist(x_session_id, "Uji Konvergensi (Iterasi #1)")
    sync_session_to_firebase(x_session_id)
    return {
        "status": "success",
        "is_converged": False,
        "iteration": 1,
        "movement": move,
        "history": history,
        "message": f"Iterasi #1 selesai. Pusat klaster masih bergerak (Movement: {move:.4f}). Silakan klik 'Jalankan Auto-Konvergensi' untuk mengulang hingga stabil."
    }


@app.post("/stepwise/auto-converge/")
@app.post("/stepwise/auto-converge")
async def auto_converge(x_session_id: Optional[str] = Header(None)):
    """
    Step 20 - Phase 2: Complete Auto Convergence Path.
    Executes iterative K-Means from init_centroids until delta < 1e-4.
    """
    session = await get_valid_session(x_session_id)
    state = session.get("algo_state", {})

    config = session.get("config", {})
    feats = state.get("features", config.get("features", list(session["df"].select_dtypes(include=['number']).columns)))
    config_ignored = config.get("ignored", [])
    protected_cols = set([config.get("identity", ""), config.get("label", "")] + (config_ignored if isinstance(config_ignored, list) else []))
    feats = [f for f in feats if f in session["df"].columns and f not in protected_cols]

    k = state.get("k", config.get("k", 3))
    ahp = config.get("ahp_weights")

    X = get_weighted_x(session["df"][feats].fillna(0).values, ahp, feats)

    init_c = state.get("init_centroids")
    if init_c and len(init_c) == k:
        centroids = np.array(init_c)
    else:
        centroids = np.array(state.get("centroids", X[:k]))
        if len(centroids) != k:
            centroids = X[:k]

    history = []
    assignments = np.zeros(len(X), dtype=int)

    for i in range(1, 101):
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        assignments = np.argmin(dists, axis=1)
        wcss = safe_float(np.sum(np.min(dists, axis=1)**2))

        new_c = np.array([X[assignments == j].mean(axis=0) if len(X[assignments == j]) > 0 else centroids[j] for j in range(k)])
        move = safe_float(np.linalg.norm(new_c - centroids))

        is_stbl = move < 1e-4 or i == 100
        history.append({
            "iter": i,
            "movement": move,
            "wcss": wcss,
            "status": "STABIL" if is_stbl else "Bergerak"
        })

        centroids = new_c
        if is_stbl and i >= 2:
            break

    # Rank-Based Reordering (C1: Berprestasi, C2: Performa Stabil, C3: Butuh Bantuan)
    new_labels, remap = reorder_clusters_by_quality(session["df"], feats, assignments, k)
    final_centroids = [centroids[old_id].tolist() for old_id, _ in sorted(remap.items(), key=lambda x: x[1])]

    session["df"]["cluster"] = new_labels.tolist()
    for j in range(k):
        session["df"][f"dist_c{j}"] = np.linalg.norm(X - np.array(final_centroids[j]), axis=1)

    eval_k = calculate_cluster_metrics(session["df"], feats, new_labels, k, ahp)

    # IF THESIS MODE: Run Iterative Fuzzy C-Means (FCM) using K-Means Final Centroids as V(0)
    analysis_mode = config.get("analysis_mode", "thesis")
    if analysis_mode == "thesis":
        m = 2.0
        centers = np.array(final_centroids)
        dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
        power = -2.0 / (m - 1)
        dists_p = dists ** power
        U = (dists_p / dists_p.sum(axis=1, keepdims=True)).T

        fcm_history = []
        for fcm_iter in range(1, 101):
            new_centers = ((U**m) @ X) / ((U**m).sum(axis=1)[:, np.newaxis] + 1e-10)
            dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - new_centers, axis=2), 1e-10)
            dists_p = dists ** power
            new_U = (dists_p / dists_p.sum(axis=1, keepdims=True)).T

            diff = safe_float(np.linalg.norm(new_U - U))
            fcm_history.append({"iter": fcm_iter, "diff": diff})

            U = new_U
            centers = new_centers
            if diff < 1e-4 and fcm_iter >= 2:
                break

        # Remap U to match reordered cluster IDs (C1, C2, C3)
        remapped_U = np.zeros_like(U)
        for old_id, new_id in remap.items():
            if old_id < len(U) and new_id < len(U):
                remapped_U[new_id] = U[old_id]

        fcm_labels = np.argmax(remapped_U, axis=0)
        session["df"]["kmeans_cluster"] = new_labels.tolist()
        session["df"]["fcm_cluster"] = fcm_labels.tolist()
        session["df"]["cluster"] = fcm_labels.tolist() # Thesis Mode final academic category comes from FCM argmax

        for j in range(k):
            session["df"][f"membership_c{j+1}"] = np.round(remapped_U[j], 4).tolist()

        xb_val = calculate_xie_beni(X, remapped_U, centers, m)
        pe_val = calculate_partition_entropy(remapped_U)

        eval_k.update({
            "fcm_iterations": len(fcm_history),
            "xie_beni_index": xb_val,
            "partition_entropy": pe_val,
            "fcm_centers": centers.tolist(),
            "fcm_history": fcm_history,
            "fcm_counts": {
                "C1_Berprestasi": int(np.sum(fcm_labels == 0)),
                "C2_Berkembang": int(np.sum(fcm_labels == 1)),
                "C3_Perlu_Pembinaan": int(np.sum(fcm_labels == 2))
            },
            "kmeans_counts": {
                "C1_Berprestasi": int(np.sum(new_labels == 0)),
                "C2_Berkembang": int(np.sum(new_labels == 1)),
                "C3_Perlu_Pembinaan": int(np.sum(new_labels == 2))
            }
        })

    session.update({
        "metrics": eval_k,
        "algo_state": {
            **state,
            "centroids": final_centroids,
            "assignments": new_labels.tolist(),
            "is_converged": True,
            "history": history,
            "iteration": len(history)
        }
    })

    add_to_checklist(x_session_id, "Riset Selesai")
    sync_session_to_firebase(x_session_id)
    return {
        "status": "success",
        "is_converged": True,
        "iteration": len(history),
        "history": history,
        "evaluation": eval_k,
        "hasil_cluster": session["df"].fillna(0).to_dict(orient="records")
    }

# --- 6. FCM CORE ---

@app.post("/stepwise/fcm-init/")
@app.post("/stepwise/fcm-init")
async def fcm_init_step(x_session_id: Optional[str] = Header(None), params: Dict[str, Any] = Body({"k": 3, "m": 2.0})):
    session = await get_valid_session(x_session_id)
    k, m = params.get("k", 3), params.get("m", 2.0)
    feats = session["config"].get("features", list(session["df"].select_dtypes(include=['number']).columns))
    protected_cols = set([session["config"].get("identity", ""), session["config"].get("label", "")] + (session["config"].get("ignored", []) if isinstance(session["config"].get("ignored", []), list) else []))
    feats = [f for f in feats if f in session["df"].columns and f not in protected_cols]

    X = get_weighted_x(session["df"][feats].fillna(0).values, session["config"].get("ahp_weights"), feats)

    state = session.get("algo_state", {})
    if "centroids" in state and len(state["centroids"]) == k:
        centers = np.array(state["centroids"])
        dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10)
        power = -2.0 / (m - 1)
        dists_p = dists ** power
        U = (dists_p / dists_p.sum(axis=1, keepdims=True)).T
        init_type = "K-Means Final Centroids"
    else:
        U = np.random.dirichlet(np.ones(k), size=len(X)).T
        init_type = "Random Dirichlet"

    session["algo_state"] = {
        "mode": "fcm",
        "iteration": 0,
        "U": U.tolist(),
        "X": X.tolist(),
        "features": feats,
        "k": k,
        "m": m,
        "history": [],
        "init_type": init_type
    }
    add_to_checklist(x_session_id, f"Inisialisasi FCM ({init_type})")
    sync_session_to_firebase(x_session_id)
    return {"status": "success", "init_type": init_type, "k": k, "m": m}

@app.post("/stepwise/fcm-iteration/")
@app.post("/stepwise/fcm-iteration")
async def fcm_iteration_step(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); state = session["algo_state"]; X, U, m, k = np.array(state["X"]), np.array(state["U"]), state["m"], state["k"]
    centers = ((U**m) @ X) / ((U**m).sum(axis=1)[:, np.newaxis] + 1e-10); dists = np.fmax(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), 1e-10); new_U = ( (dists ** (-2.0 / (m - 1))) / (dists ** (-2.0 / (m - 1))).sum(axis=1)[:, np.newaxis] ).T
    diff = safe_float(np.linalg.norm(new_U - U)); state.update({"U": new_U.tolist(), "centroids": centers.tolist(), "iteration": state["iteration"] + 1})
    if diff < 1e-4: labels = np.argmax(new_U, axis=0); session["df"]["cluster"] = labels.tolist(); eval_f = calculate_cluster_metrics(session["df"], state["features"], labels, k, session["config"].get("ahp_weights")); session.update({"metrics": eval_f})
    return {"status": "success", "is_converged": diff < 1e-4, "diff": diff}

# --- 7. AUDITS & VISUALS ---

@app.post("/stepwise/stability-audit/")
@app.post("/stepwise/stability-audit")
async def stability_audit(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); feats = session["config"]["features"]; X = get_weighted_x(session["df"][feats].fillna(0).values, session["config"].get("ahp_weights"), feats)
    return perform_stability_audit(X, session["config"].get("k", 3), session["df"]["cluster"].values)

@app.post("/stepwise/sensitivity-audit/")
@app.post("/stepwise/sensitivity-audit")
async def sensitivity_audit(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); feats = session["config"]["features"]; X_raw = session["df"][feats].fillna(0).values
    return perform_sensitivity_audit(X_raw, feats, session["config"].get("ahp_weights"), session["config"].get("k", 3), session["df"]["cluster"].values)

@app.get("/stepwise/spatial-map/")
@app.get("/stepwise/spatial-map")
async def spatial_map(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); from sklearn.decomposition import PCA; feats = session["config"]["features"]; X = get_weighted_x(session["df"][feats].fillna(0).values, session["config"].get("ahp_weights"), feats)
    pca = PCA(n_components=2); X_2d = pca.fit_transform(X); data = [{"x": safe_float(X_2d[i, 0]), "y": safe_float(X_2d[i, 1]), "cluster": int(session["df"]["cluster"].values[i]), "label": str(session["df"][session["config"].get("label", "nama")].values[i])} for i in range(len(X_2d))]
    loadings = [[{"feature": f, "loading": safe_float(pca.components_[0, i])} for i, f in enumerate(feats)], [{"feature": f, "loading": safe_float(pca.components_[1, i])} for i, f in enumerate(feats)]]
    return {"status": "success", "data": data, "total_variance": safe_float(np.sum(pca.explained_variance_ratio_)), "loadings": loadings}

@app.get("/stepwise/final-analysis/")
@app.get("/stepwise/final-analysis")
async def get_final_analysis(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); m = session.get("metrics", {})
    return {"status": "success", "jumlah_data": len(session["df"]), "centroids": session.get("algo_state", {}).get("centroids", []), "config": session.get("config", {}), "metrics": m, "hasil_cluster": session["df"].fillna(0).to_dict(orient="records")}

@app.get("/stepwise/export-excel/")
@app.get("/stepwise/export-excel")
async def export_excel_route(x_session_id: Optional[str] = Header(None)):
    session = await get_valid_session(x_session_id); output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_exp = session["df"].copy(); df_exp["cluster"] = df_exp["cluster"] + 1; df_exp.to_excel(writer, sheet_name="Hasil_Final", index=False)
    output.seek(0); return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.get("/stepwise/explain-siswa/")
@app.get("/stepwise/explain-siswa")
async def explain_siswa(x_session_id: Optional[str] = Header(None), nis: str = ""):
    session = await get_valid_session(x_session_id); id_col = session["config"].get("identity", "nis")
    student = session["df"][session["df"][id_col].astype(str) == str(nis)]
    if student.empty: raise HTTPException(404, "Siswa not found")
    idx = int(student["cluster"].values[0]); target = np.array(session["metrics"]["centroids"])[idx]
    feats = session["config"]["features"]; X_s = student[feats].fillna(0).values[0]
    ranges = np.ptp(session["df"][feats].fillna(0).values, axis=0) + 1e-10
    contribs = [{"feature": f, "val": 1.0 - (abs(X_s[i] - target[i]) / ranges[i])} for i, f in enumerate(feats)]
    return {"status": "success", "contributions": contribs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
