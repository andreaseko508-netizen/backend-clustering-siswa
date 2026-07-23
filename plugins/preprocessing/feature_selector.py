import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sdk.base_plugin import BaseResearchPlugin
from sdk.models import ExecutionContext, ExecutionResult, ResearchArtifact
import os

class FeatureSelectorPlugin(BaseResearchPlugin):
    def get_plugin_id(self) -> str:
        return "preprocessing.feature_selector.FeatureSelectorPlugin"

    def get_name(self) -> str:
        return "Feature Selector"

    def execute(self, context: ExecutionContext) -> ExecutionResult:
        try:
            dataset_path = context.input_datasets.get("primary")
            df = pd.read_csv(dataset_path)

            # 1. Select numeric features
            numeric_df = df.select_dtypes(include=[np.number])

            # 2. Variance Threshold
            threshold = float(context.parameters.get("variance_threshold", 0.0))
            selector = VarianceThreshold(threshold=threshold)
            selected_data = selector.fit_transform(numeric_df)

            # Get selected column names
            selected_cols = numeric_df.columns[selector.get_support()].tolist()
            final_df = df[selected_cols]

            # 3. Save Artifact
            result_file = os.path.join(context.artifact_path, "selected_features.csv")
            os.makedirs(context.artifact_path, exist_ok=True)
            final_df.to_csv(result_file, index=False)

            return ExecutionResult(
                status="SUCCESS",
                metrics={
                    "original_feature_count": len(numeric_df.columns),
                    "selected_feature_count": len(selected_cols),
                    "removed_features": list(set(numeric_df.columns) - set(selected_cols))
                },
                artifacts=[
                    ResearchArtifact(
                        name="Selected Features Dataset",
                        type="DATASET_CSV",
                        file_path=result_file,
                        metadata={"columns": selected_cols}
                    )
                ]
            )
        except Exception as e:
            return ExecutionResult(status="FAILED", metrics={}, artifacts=[], error_message=str(e))
