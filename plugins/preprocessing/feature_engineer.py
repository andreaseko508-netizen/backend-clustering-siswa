import pandas as pd
import numpy as np
from sdk.base_plugin import BaseResearchPlugin
from sdk.models import ExecutionContext, ExecutionResult, ResearchArtifact
import os

class FeatureEngineeringPlugin(BaseResearchPlugin):
    def get_plugin_id(self) -> str:
        return "preprocessing.feature_engineer.FeatureEngineeringPlugin"

    def get_name(self) -> str:
        return "Feature Engineer"

    def execute(self, context: ExecutionContext) -> ExecutionResult:
        try:
            dataset_path = context.input_datasets.get("primary")
            df = pd.read_csv(dataset_path)

            ops = context.parameters.get("operations", [])
            new_cols = []

            for op in ops:
                col = op.get("column")
                action = op.get("action")

                if action == "log" and col in df.columns:
                    df[f"{col}_log"] = np.log1p(df[col])
                    new_cols.append(f"{col}_log")
                elif action == "binning" and col in df.columns:
                    bins = op.get("bins", 5)
                    df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=False)
                    new_cols.append(f"{col}_binned")

            # Save
            result_file = os.path.join(context.artifact_path, "engineered_features.csv")
            os.makedirs(context.artifact_path, exist_ok=True)
            df.to_csv(result_file, index=False)

            return ExecutionResult(
                status="SUCCESS",
                metrics={"engineered_columns_count": len(new_cols)},
                artifacts=[
                    ResearchArtifact(
                        name="Engineered Dataset",
                        type="DATASET_CSV",
                        file_path=result_file,
                        metadata={"new_columns": new_cols}
                    )
                ]
            )
        except Exception as e:
            return ExecutionResult(status="FAILED", metrics={}, artifacts=[], error_message=str(e))
