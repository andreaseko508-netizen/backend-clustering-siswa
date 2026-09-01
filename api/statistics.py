import numpy as np
import pandas as pd
import time
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2, ttest_rel, wilcoxon, shapiro, skew
from sklearn.metrics import adjusted_rand_score

# S2 PROFESSIONAL HELPERS
def safe_float(val):
    try:
        if val is None or np.isnan(val) or np.isinf(val): return 0.0
        return float(val)
    except: return 0.0

def calculate_cluster_metrics(df, features, assignments, k, weights_dict=None):
    """Professional Grade Evaluation Metrics for Research."""
    try:
        X_raw = df[features].select_dtypes(include=[np.number]).fillna(0).values
        if weights_dict:
            w = np.array([weights_dict.get(f, 1.0) for f in features])
            X = X_raw * np.sqrt(w)
        else:
            X = X_raw

        unique_labels = np.unique(assignments)
        if len(unique_labels) < 2:
            return {"status": "error", "message": "Hanya ditemukan 1 klaster. Data tidak cukup variatif."}

        dbi = safe_float(davies_bouldin_score(X, assignments))
        sil = safe_float(silhouette_score(X, assignments))
        chi = safe_float(calinski_harabasz_score(X, assignments))

        # 1. Silhouette Samples for Plotting
        sample_sil_values = silhouette_samples(X, assignments)
        silhouette_values = []
        for i in range(k):
            ith_cluster_sil_values = sample_sil_values[assignments == i]
            ith_cluster_sil_values.sort()
            silhouette_values.append({
                "cluster": int(i),
                "values": [safe_float(v) for v in ith_cluster_sil_values],
                "avg": safe_float(np.mean(ith_cluster_sil_values)) if len(ith_cluster_sil_values) > 0 else 0.0
            })

        # 2. WCSS (Density Analysis)
        wcss = 0.0
        for i in range(k):
            cluster_points = X[assignments == i]
            if len(cluster_points) > 0:
                center = cluster_points.mean(axis=0)
                wcss += np.sum((cluster_points - center)**2)

        # 3. Cluster Profiling (Centroid Characterization)
        dist = {str(i): {"count": int(np.sum(assignments == i)), "percentage": safe_float(np.sum(assignments == i) / len(df) * 100)} for i in range(k)}
        profiles = {str(i): {f: safe_float(v) for f, v in df[assignments == i][features].mean(numeric_only=True).fillna(0).to_dict().items()} for i in range(k)}

        # 4. Intelligence Advice (XAI Integration)
        improvement_advice = []
        if sil < 0.4: improvement_advice.append("Koefisien Silhouette rendah (< 0.4). Struktur kelompok kurang kuat.")
        if dbi > 1.2: improvement_advice.append("Indeks Davies-Bouldin tinggi (> 1.2). Kelompok terlalu berdekatan (overlapping).")

        return {
            "davies_bouldin_index": dbi,
            "silhouette_score": sil,
            "calinski_harabasz_index": chi,
            "wcss": safe_float(wcss),
            "distribution": dist,
            "cluster_profiles": profiles,
            "improvement_advice": improvement_advice,
            "silhouette_plot_data": silhouette_values,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_real_ga_init(X, k, population_size=40, generations=50):
    """
    PROFESSIONAL GENETIC ALGORITHM FOR CENTROID DISCOVERY.
    Eliminates Local Optima traps for critical perbatasan datasets.
    """
    n_samples, n_features = X.shape

    def calculate_fitness(centroids):
        # Fitness = 1 / (Total Distance + 1e-10)
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        min_dists = np.min(dists, axis=1)
        wcss = np.sum(min_dists**2)
        return 1.0 / (wcss + 1e-10)

    # Initial Population: Random samples from X
    population = [X[np.random.choice(n_samples, k, replace=False)] for _ in range(population_size)]

    for gen in range(generations):
        fitness_scores = [calculate_fitness(ind) for ind in population]

        # Selection: Tournament
        new_population = []
        for _ in range(population_size):
            i1, i2 = np.random.choice(population_size, 2, replace=False)
            winner = population[i1] if fitness_scores[i1] > fitness_scores[i2] else population[i2]
            new_population.append(winner.copy())

        # Crossover & Mutation
        for i in range(0, population_size, 2):
            if np.random.rand() < 0.8: # Crossover rate
                mix_point = np.random.randint(1, k)
                new_population[i][:mix_point], new_population[i+1][:mix_point] = \
                    new_population[i+1][:mix_point].copy(), new_population[i][:mix_point].copy()

            # Mutation: Jitter a centroid slightly
            if np.random.rand() < 0.2: # Mutation rate
                m_idx = np.random.randint(k)
                new_population[i][m_idx] += np.random.normal(0, 0.05, n_features)

        population = new_population

    # Return best individual
    best_idx = np.argmax([calculate_fitness(ind) for ind in population])
    return population[best_idx]

def perform_normality_test_expert(df, features):
    """S2 Standard: Detailed Normality & Skewness Audit."""
    results = []
    for f in features:
        data = df[f].fillna(0).values
        if len(data) > 3:
            stat, p = shapiro(data)
            sk = skew(data)
            is_normal = p > 0.05
            results.append({
                "feature": f,
                "p_value": safe_float(p),
                "skewness": safe_float(sk),
                "is_normal": bool(is_normal),
                "risk": "HIGH" if abs(sk) > 2 else "LOW"
            })

    # Methodological recommendation
    non_normal_count = sum(1 for r in results if not r["is_normal"])
    recommendation = "RobustScaler (Pilihan Tepat)" if non_normal_count > (len(features)/2) else "StandardScaler (Aman)"

    return {
        "results": results,
        "recommendation": recommendation,
        "non_normal_count": non_normal_count,
        "justification": f"Ditemukan {non_normal_count} variabel dengan distribusi tidak normal."
    }

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
        aw = matrix @ weights
        λ_max = np.mean(aw / np.where(weights == 0, 1e-10, weights))
        ci = (λ_max - n) / (n - 1) if n > 1 else 0
        ri_table = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
        cr = ci / ri_table.get(n, 1.49) if ri_table.get(n, 1.49) > 0 else 0
        return np.nan_to_num(weights), safe_float(cr)
    except: return np.ones(len(matrix))/len(matrix), 0.0

def get_weighted_x(X, weights_dict, features):
    if not weights_dict: return X
    w = np.array([weights_dict.get(f, 1.0) for f in features])
    return X * np.sqrt(w)
