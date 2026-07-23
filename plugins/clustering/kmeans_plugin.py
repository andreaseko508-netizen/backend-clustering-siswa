import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import time
from sdk.base_plugin import BaseResearchPlugin
from sdk.models import ExecutionContext, ExecutionResult, ResearchArtifact
from sdk.explainability import PrescriptiveExplainabilityEngine

class KMeansPlugin(BaseResearchPlugin):
    def get_plugin_id(self) -> str:
        return "clustering.kmeans_plugin.KMeansPlugin"

    def get_name(self) -> str:
        return "K-Means Clustering"

    def execute(self, context: ExecutionContext) -> ExecutionResult:
        try:
            dataset_path = context.input_datasets.get("primary")
            df = pd.read_csv(dataset_path)
            X = df.select_dtypes(include=[np.number]).fillna(0)

            n_clusters = int(context.parameters.get("n_clusters", 3))
            random_state = int(context.parameters.get("random_seed", 42))

            if len(X) < n_clusters:
                return ExecutionResult(status="FAILED", metrics={}, artifacts=[], error_message=f"Dataset too small for K={n_clusters}")

            start_time = time.time()
            data_values = X.values
            n_samples, n_features = data_values.shape

            # Initial Centroids
            np.random.seed(random_state)
            init_indices = np.random.choice(n_samples, n_clusters, replace=False)
            centroids = data_values[init_indices].copy()

            clustering_checkpoints = {
                "Centroid Awal": centroids.tolist(),
                "Jarak Euclidean Awal": None, # Will fill in first iter
                "Pembagian Cluster Awal": None
            }

            # Identify a sample student for UI education (e.g., index 0)
            sample_student_data = df.iloc[0].to_dict()
            sample_student_vals = data_values[0]

            history = []
            max_iterations = 100

            for i in range(max_iterations):
                # 1. Assignment Step
                diff = data_values[:, np.newaxis] - centroids
                sq_diff = diff ** 2
                sum_sq_diff = np.sum(sq_diff, axis=2)
                dists = np.sqrt(sum_sq_diff)
                labels = np.argmin(dists, axis=1)

                # Capture sample student's distances for this iteration
                sample_dists = dists[0].tolist()

                if i == 0:
                    clustering_checkpoints["Jarak Euclidean Awal"] = dists.head(10).tolist() if hasattr(dists, 'head') else dists[:10].tolist()
                    clustering_checkpoints["Pembagian Cluster Awal"] = labels[:100].tolist()

                iter_record = {
                    "iteration": i + 1,
                    "centroids_before": centroids.tolist(),
                    "cluster_counts": [int(np.sum(labels == j)) for j in range(n_clusters)],
                    "sample_student": {
                        "name": str(df.iloc[0].get("nama", "Student 1")),
                        "values": sample_student_vals.tolist(),
                        "distances": sample_dists,
                        "assigned_cluster": int(labels[0])
                    }
                }

                # 2. Update Step
                new_centroids = np.zeros_like(centroids)
                for j in range(n_clusters):
                    cluster_members = data_values[labels == j]
                    if len(cluster_members) > 0:
                        new_centroids[j] = cluster_members.mean(axis=0)
                    else:
                        new_centroids[j] = centroids[j]

                delta = float(np.sum(np.sqrt(np.sum((new_centroids - centroids)**2, axis=1))))
                iter_record["centroids_after"] = new_centroids.tolist()
                iter_record["delta_movement"] = delta
                history.append(iter_record)

                if np.allclose(centroids, new_centroids, atol=1e-4):
                    break
                centroids = new_centroids

            end_time = time.time()
            df['cluster'] = labels
            for i in range(n_clusters):
                df[f'dist_c{i}'] = dists[:, i]

            # Final Evaluation
            sil = float(silhouette_score(data_values, labels))
            dbi = float(davies_bouldin_score(data_values, labels))
            chi = float(calinski_harabasz_score(data_values, labels))
            wcss = float(np.sum((data_values - centroids[labels])**2))

            metrics = {
                "silhouette_score": sil,
                "davies_bouldin_index": dbi,
                "calinski_harabasz_index": chi,
                "wcss": wcss,
                "n_clusters": n_clusters,
                "iterations": len(history),
                "runtime_sec": float(end_time - start_time),
                "centroids": centroids.tolist(),
                "feature_names": X.columns.tolist(),
                "cluster_profiles": {str(j): X[labels == j].mean().to_dict() for j in range(n_clusters)},
                "iteration_history": history,
                "clustering_checkpoints": clustering_checkpoints
            }

            result_file = os.path.join(context.artifact_path, "kmeans_result.csv")
            os.makedirs(context.artifact_path, exist_ok=True)
            df.to_csv(result_file, index=False)

            return ExecutionResult(status="SUCCESS", metrics=metrics, artifacts=[ResearchArtifact(name="K-Means Result", type="DATASET_CSV", file_path=result_file)])
        except Exception as e:
            return ExecutionResult(status="FAILED", metrics={}, artifacts=[], error_message=str(e))
