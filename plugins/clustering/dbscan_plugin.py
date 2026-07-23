import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import os
from sdk.base_plugin import BaseResearchPlugin
from sdk.models import ExecutionContext, ExecutionResult, ResearchArtifact

class DBSCANPlugin(BaseResearchPlugin):
    def get_plugin_id(self) -> str:
        return "clustering.dbscan_plugin.DBSCANPlugin"

    def get_name(self) -> str:
        return "DBSCAN Clustering"

    def execute(self, context: ExecutionContext) -> ExecutionResult:
        try:
            dataset_path = context.input_datasets.get("primary")
            df = pd.read_csv(dataset_path)

            eps = float(context.parameters.get("eps", 0.5))
            min_samples = int(context.parameters.get("min_samples", 5))

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            X = df.select_dtypes(include=[np.number])
            labels = dbscan.fit_predict(X)

            df['cluster'] = labels

            # Metrics (Silhouette only if more than 1 cluster)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            metrics = {
                "n_clusters": n_clusters,
                "n_noise_points": int(list(labels).count(-1))
            }

            if n_clusters > 1:
                metrics["silhouette_score"] = float(silhouette_score(X, labels))

            # Save
            result_file = os.path.join(context.artifact_path, "dbscan_result.csv")
            os.makedirs(context.artifact_path, exist_ok=True)
            df.to_csv(result_file, index=False)

            return ExecutionResult(
                status="SUCCESS",
                metrics=metrics,
                artifacts=[
                    ResearchArtifact(name="DBSCAN Result", type="DATASET_CSV", file_path=result_file)
                ]
            )
        except Exception as e:
            return ExecutionResult(status="FAILED", metrics={}, artifacts=[], error_message=str(e))
