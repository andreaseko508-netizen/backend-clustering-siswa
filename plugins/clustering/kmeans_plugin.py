import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import time
from sdk.base_plugin import BaseResearchPlugin
from sdk.models import ExecutionContext, ExecutionResult, ResearchArtifact

class KMeansPlugin(BaseResearchPlugin):
    def get_plugin_id(self) -> str:
        return "clustering.kmeans_plugin.KMeansPlugin"

    def get_name(self) -> str:
        return "K-Means Clustering"

    def _kmeans_plus_plus(self, data, k, weights=None):
        """Standard K-Means++ Initialization Strategy for Research Reliability."""
        n_samples, n_features = data.shape
        centroids = np.zeros((k, n_features))

        # 1. Randomly pick first centroid
        idx = np.random.randint(n_samples)
        centroids[0] = data[idx]

        for i in range(1, k):
            # 2. Compute squared distances to nearest already chosen centroid
            diff = data[:, np.newaxis] - centroids[:i]
            if weights is not None:
                sq_diff = (diff ** 2) * weights
            else:
                sq_diff = diff ** 2

            # Sum over features, then take min over chosen centroids
            min_dists_sq = np.min(np.sum(sq_diff, axis=2), axis=1)

            # 3. Pick next centroid with probability proportional to D(x)^2
            sum_sq = np.sum(min_dists_sq)
            if sum_sq == 0:
                probs = np.ones(n_samples) / n_samples
            else:
                probs = min_dists_sq / sum_sq

            next_idx = np.random.choice(n_samples, p=probs)
            centroids[i] = data[next_idx]

        return centroids

    def _run_single_kmeans(self, data, k, max_iter, weights=None, init_centroids=None):
        """Internal K-Means solver with Weighted Euclidean support."""
        n_samples, n_features = data.shape
        centroids = init_centroids.copy()

        history = []
        labels = None
        dists = None

        for i in range(max_iter):
            # 1. Assignment Step
            diff = data[:, np.newaxis] - centroids
            if weights is not None:
                sq_diff = (diff ** 2) * weights
            else:
                sq_diff = diff ** 2

            dists_sq = np.sum(sq_diff, axis=2)
            dists = np.sqrt(dists_sq)
            labels = np.argmin(dists, axis=1)

            # 2. Update Step
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                cluster_members = data[labels == j]
                if len(cluster_members) > 0:
                    new_centroids[j] = cluster_members.mean(axis=0)
                else:
                    # Keep old centroid if no points assigned (rare with kmeans++)
                    new_centroids[j] = centroids[j]

            delta = float(np.sum(np.sqrt(np.sum((new_centroids - centroids)**2, axis=1))))

            iter_record = {
                "iteration": i + 1,
                "wcss": float(np.sum(np.min(dists_sq, axis=1))),
                "delta_movement": delta
            }
            history.append(iter_record)

            if np.allclose(centroids, new_centroids, atol=1e-5):
                break
            centroids = new_centroids

        final_wcss = history[-1]["wcss"]
        return labels, centroids, dists, history, final_wcss

    def execute(self, context: ExecutionContext) -> ExecutionResult:
        try:
            dataset_path = context.input_datasets.get("primary")
            if not dataset_path:
                return ExecutionResult(status="FAILED", metrics={}, artifacts=[], error_message="Primary dataset missing")

            df = pd.read_csv(dataset_path)
            X_all = df.select_dtypes(include=[np.number]).fillna(0)

            # Use specific features if provided, otherwise use all numeric
            features = context.parameters.get("features")
            if features:
                X = X_all[features]
            else:
                X = X_all

            n_clusters = int(context.parameters.get("n_clusters", 3))
            random_state = int(context.parameters.get("random_seed", 42))
            n_init = int(context.parameters.get("n_init", 10)) # Multi-restart for S2 Standard
            max_iter = int(context.parameters.get("max_iter", 100))

            # Dynamic Weights Integration
            weights_input = context.parameters.get("weights")
            weights = None
            if weights_input:
                # Map dict weights to array aligned with X columns
                weights = np.array([weights_input.get(col, 1.0) for col in X.columns])

            if len(X) < n_clusters:
                return ExecutionResult(status="FAILED", metrics={}, artifacts=[], error_message=f"Dataset size ({len(X)}) is less than K ({n_clusters})")

            np.random.seed(random_state)
            start_time = time.time()
            data_values = X.values

            best_wcss = float('inf')
            best_labels = None
            best_centroids = None
            best_dists = None
            best_history = None

            # Multi-Restart Optimization Loop
            for _ in range(n_init):
                init_centroids = self._kmeans_plus_plus(data_values, n_clusters, weights)
                labels, centroids, dists, history, wcss = self._run_single_kmeans(data_values, n_clusters, max_iter, weights, init_centroids)

                if wcss < best_wcss:
                    best_wcss = wcss
                    best_labels = labels
                    best_centroids = centroids
                    best_dists = dists
                    best_history = history

            end_time = time.time()

            # Scientific Evaluation Metrics
            sil = float(silhouette_score(data_values, best_labels))
            dbi = float(davies_bouldin_score(data_values, best_labels))
            chi = float(calinski_harabasz_score(data_values, best_labels))

            metrics = {
                "silhouette_score": sil,
                "davies_bouldin_index": dbi,
                "calinski_harabasz_index": chi,
                "wcss": best_wcss,
                "n_clusters": n_clusters,
                "iterations": len(best_history),
                "runtime_sec": float(end_time - start_time),
                "centroids": best_centroids.tolist(),
                "feature_names": X.columns.tolist(),
                "cluster_profiles": {str(j): X[best_labels == j].mean().to_dict() for j in range(n_clusters)},
                "iteration_history": best_history,
                "initialization_strategy": "kmeans++",
                "multi_init_count": n_init
            }

            # Prepare Artifacts
            df['cluster'] = best_labels
            for i in range(n_clusters):
                df[f'dist_c{i}'] = best_dists[:, i]

            result_file = os.path.join(context.artifact_path, "kmeans_result.csv")
            os.makedirs(context.artifact_path, exist_ok=True)
            df.to_csv(result_file, index=False)

            return ExecutionResult(
                status="SUCCESS",
                metrics=metrics,
                artifacts=[ResearchArtifact(name="K-Means Result", type="DATASET_CSV", file_path=result_file)]
            )
        except Exception as e:
            import traceback
            return ExecutionResult(status="FAILED", metrics={}, artifacts=[], error_message=f"{str(e)}\n{traceback.format_exc()}")
