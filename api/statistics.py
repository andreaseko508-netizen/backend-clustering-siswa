import numpy as np
import pandas as pd
import time
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2, ttest_rel, wilcoxon, shapiro
from sklearn.metrics import adjusted_rand_score

# Try relative and absolute imports for flexibility
try:
    from api.utils import safe_float
except ImportError:
    try:
        from utils import safe_float
    except ImportError:
        def safe_float(val):
            try:
                if val is None or np.isnan(val) or np.isinf(val): return 0.0
                return float(val)
            except: return 0.0

def calculate_cluster_metrics(df, features, assignments, k, weights_dict=None):
    try:
        X_raw = df[features].select_dtypes(include=[np.number]).fillna(0).values
        if weights_dict:
            w = np.array([weights_dict.get(f, 1.0) for f in features])
            X = X_raw * np.sqrt(w)
        else:
            X = X_raw

        unique_labels = np.unique(assignments)
        dbi = safe_float(davies_bouldin_score(X, assignments)) if len(unique_labels) > 1 else 0.0
        sil = safe_float(silhouette_score(X, assignments)) if len(unique_labels) > 1 else 0.0
        chi = safe_float(calinski_harabasz_score(X, assignments)) if len(unique_labels) > 1 else 0.0

        silhouette_values = []
        if len(unique_labels) > 1:
            sample_sil_values = silhouette_samples(X, assignments)
            for i in range(k):
                ith_cluster_sil_values = sample_sil_values[assignments == i]
                ith_cluster_sil_values.sort()
                silhouette_values.append({
                    "cluster": int(i),
                    "values": [safe_float(v) for v in ith_cluster_sil_values],
                    "avg": safe_float(np.mean(ith_cluster_sil_values)) if len(ith_cluster_sil_values) > 0 else 0.0
                })

        wcss = 0.0
        if len(unique_labels) > 1:
            for i in range(k):
                cluster_points = X[assignments == i]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    wcss += np.sum((cluster_points - center)**2)

        dist = {str(i): {"count": int(np.sum(assignments == i)), "percentage": safe_float(np.sum(assignments == i) / len(df) * 100)} for i in range(k)}
        profiles = {str(i): {f: safe_float(v) for f, v in df[assignments == i][features].mean(numeric_only=True).fillna(0).to_dict().items()} for i in range(k)}

        centroid_matrix = np.array([profiles[str(i)].get(f, 0) for i in range(k) for f in features]).reshape(k, -1)
        centroid_matrix = np.nan_to_num(centroid_matrix)
        variances = np.var(centroid_matrix, axis=0)
        importance_sum = np.sum(variances) if np.sum(variances) > 0 else 1.0
        feature_importance = {f: safe_float((v / importance_sum) * 100) for f, v in zip(features, variances)}

        corr_matrix = pd.DataFrame(X_raw, columns=features).corr().abs()
        redundant_features = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if corr_matrix.iloc[i, j] > 0.90:
                    redundant_features.append({"f1": features[i], "f2": features[j], "val": safe_float(corr_matrix.iloc[i, j])})

        improvement_advice = []
        if sil < 0.35: improvement_advice.append("Koefisien Silhouette rendah.")
        if dbi > 1.0: improvement_advice.append("Indeks Davies-Bouldin tinggi (> 1.0).")
        if redundant_features: improvement_advice.append(f"Ditemukan {len(redundant_features)} variabel redundan.")

        return {
            "davies_bouldin_index": dbi,
            "silhouette_score": sil,
            "calinski_harabasz_index": chi,
            "wcss": safe_float(wcss),
            "distribution": dist,
            "cluster_profiles": profiles,
            "feature_importance": feature_importance,
            "improvement_advice": improvement_advice,
            "silhouette_plot_data": silhouette_values,
            "dbi": dbi,
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"Metrics Error: {e}")
        return {"davies_bouldin_index": 0.0, "silhouette_score": 0.0, "wcss": 0.0, "distribution": {}, "cluster_profiles": {}}

def calculate_xie_beni(X, U, centers, m):
    try:
        n_samples = X.shape[0]
        dists_sq = np.sum((X[:, np.newaxis] - centers)**2, axis=2)
        numerator = np.sum((U**m).T * dists_sq)
        centers_dist_sq = np.sum((centers[:, np.newaxis] - centers)**2, axis=2)
        np.fill_diagonal(centers_dist_sq, np.inf)
        min_sep = np.min(centers_dist_sq)
        xb = numerator / (n_samples * min_sep + 1e-10)
        return safe_float(xb)
    except: return 0.0

def calculate_partition_entropy(U):
    try:
        n_samples = U.shape[1]
        U_safe = np.fmax(U, 1e-10)
        pe = -np.sum(U * np.log(U_safe)) / n_samples
        return safe_float(pe)
    except: return 0.0

def calculate_hopkins(X):
    try:
        if X.shape[0] < 2: return 0.5
        neigh = NearestNeighbors(n_neighbors=2).fit(X)
        u_distances, _ = neigh.kneighbors(np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), X.shape), n_neighbors=1)
        w_distances, _ = neigh.kneighbors(X, n_neighbors=2)
        u_sum, w_sum = np.sum(u_distances), np.sum(w_distances[:, 1])
        return safe_float(u_sum / (u_sum + w_sum))
    except: return 0.5

def calculate_ahp_weights_and_cr(matrix):
    try:
        n = len(matrix)
        col_sum = np.sum(matrix, axis=0)
        col_sum = np.where(col_sum == 0, 1e-10, col_sum)
        norm_matrix = matrix / col_sum
        weights = np.mean(norm_matrix, axis=1)
        weights_safe = np.where(weights == 0, 1e-10, weights)
        aw = matrix @ weights
        λ_max = np.mean(aw / weights_safe)
        ci = (λ_max - n) / (n - 1) if n > 1 else 0
        ri_table = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
        cr = ci / ri_table.get(n, 1.49) if ri_table.get(n, 1.49) > 0 else 0
        return np.nan_to_num(weights), safe_float(cr)
    except: return np.ones(len(matrix))/len(matrix), 0.0

def get_weighted_x(X, weights_dict, features):
    if not weights_dict: return X
    w = np.array([weights_dict.get(f, 1.0) for f in features])
    return X * np.sqrt(w)

def perform_significance_test(X, labels_a, labels_b):
    try:
        scores_a = silhouette_samples(X, labels_a)
        scores_b = silhouette_samples(X, labels_b)
        t_stat, p_val = ttest_rel(scores_a, scores_b)
        diff = scores_a - scores_b
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
        is_significant = bool(p_val < 0.05)
        return {
            "p_value": safe_float(p_val),
            "t_statistic": safe_float(t_stat),
            "cohen_d": safe_float(d),
            "is_significant": is_significant,
            "interpretation": "Perbedaan performa SIGNIFIKAN secara statistik (p < 0.05)." if is_significant else "Perbedaan performa TIDAK SIGNIFIKAN (Hanya kebetulan)."
        }
    except Exception as e:
        return {"p_value": 1.0, "is_significant": False, "interpretation": "Gagal menghitung signifikansi."}

def perform_stability_audit(X, k, full_labels, iterations=15):
    from sklearn.cluster import KMeans
    ari_scores = []
    for i in range(iterations):
        sample_indices = np.random.choice(len(X), int(0.85 * len(X)), replace=False)
        X_sub = X[sample_indices]
        ref_labels_sub = full_labels[sample_indices]
        km_sub = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=i).fit(X_sub)
        ari = adjusted_rand_score(ref_labels_sub, km_sub.labels_)
        ari_scores.append(safe_float(ari))
    avg_stability = safe_float(np.mean(ari_scores))
    level = "EXCELLENT" if avg_stability > 0.8 else ("STABLE" if avg_stability > 0.6 else "WEAK")
    return {"stability_score": avg_stability, "level": level, "scores": ari_scores}

def perform_sensitivity_audit(X_raw, features, ahp_weights, k, original_labels):
    from sklearn.cluster import KMeans
    results = []
    for feature in features:
        ari_scores = []
        for shift in [1.1, 0.9]:
            tweaked = ahp_weights.copy()
            tweaked[feature] *= shift
            total = sum(tweaked.values())
            tweaked = {k: v/total for k, v in tweaked.items()}
            X_tweaked = X_raw * np.sqrt(np.array([tweaked.get(f, 1.0) for f in features]))
            km = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=42).fit(X_tweaked)
            ari_scores.append(safe_float(adjusted_rand_score(original_labels, km.labels_)))
        avg_ari = np.mean(ari_scores)
        results.append({"feature": feature, "stability_score": safe_float(avg_ari), "level": "Robust" if avg_ari > 0.8 else "Sensitive"})
    return {"overall_stability": safe_float(np.mean([r["stability_score"] for r in results])), "results": results}

def perform_normality_test(df, features):
    results = []
    non_normal = 0
    for f in features:
        data = df[f].fillna(0).values
        if len(data) > 3:
            stat, p = shapiro(data)
            is_normal = p > 0.05
            if not is_normal: non_normal += 1
            results.append({"feature": f, "p_value": safe_float(p), "is_normal": bool(is_normal)})
    recommendation = "RobustScaler" if non_normal > (len(features)/2) else "StandardScaler"
    return {"results": results, "recommendation": recommendation, "non_normal_count": non_normal}
