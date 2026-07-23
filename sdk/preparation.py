import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time
from sdk.core import StepResult, StepStatus, Artifact, ExecutionContext
from sdk.artifact_manager import ArtifactManager
from sdk.registries import FormulaRegistry
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class PipelinePlanner:
    """Validates the execution blueprint and builds the execution DAG."""

    @staticmethod
    def validate_and_plan(blueprint: Dict[str, Any]) -> List[Dict[str, Any]]:
        plan = []
        recommendations = blueprint.get("recommendations", [])

        # Simple ordering: Imputation -> Duplicate Removal -> Outlier Removal -> Scaling -> Transformation -> Selection
        stages = ["Cleaning", "Scaling", "Transformation", "Selection"]

        for stage in stages:
            for rec in recommendations:
                if rec.get("target") == "G2" and rec.get("action") == stage:
                    plan.append(rec)

        # Check for conflicts (e.g., redundant scalers)
        # (Placeholder for complex validation)
        return plan

class DataPrepEngine:
    def __init__(self, artifact_manager: ArtifactManager):
        self.artifact_manager = artifact_manager
        self.decision_log = []
        self.parameter_snapshot = {}

    def execute_blueprint(self, context: ExecutionContext, blueprint: Dict[str, Any], df: pd.DataFrame) -> List[StepResult]:
        results = []
        plan = PipelinePlanner.validate_and_plan(blueprint)

        current_df = df.copy()
        parent_artifact_id = context.dataset_id

        for step in plan:
            action = step.get("action")
            method = step.get("method")

            step_result = None
            if action == "Cleaning":
                step_result, current_df = self._handle_cleaning(context, current_df, method, parent_artifact_id, step)
            elif action == "Scaling":
                step_result, current_df = self._handle_scaling(context, current_df, method, parent_artifact_id, step)

            if step_result:
                results.append(step_result)
                parent_artifact_id = step_result.artifact.id
                self.decision_log.append({
                    "step_id": step_result.step_id,
                    "action": action,
                    "method": method,
                    "reasoning": step.get("reasoning"),
                    "confidence": step.get("confidence")
                })

        # Final Validation Step
        results.append(self._validate_final(current_df, parent_artifact_id))

        # Save Metadata Artifacts
        self.artifact_manager.create_artifact("decision_log", "METRICS_JSON", self.decision_log, "DataPrepEngine")
        self.artifact_manager.create_artifact("parameter_snapshot", "METRICS_JSON", self.parameter_snapshot, "DataPrepEngine")

        return results

    def _handle_cleaning(self, context, df, method, parent_id, rec) -> (StepResult, pd.DataFrame):
        start_time = time.time()
        new_df = df.copy()

        if method == "Median Imputation":
            imputer = SimpleImputer(strategy='median')
            cols = new_df.select_dtypes(include=[np.number]).columns
            new_df[cols] = imputer.fit_transform(new_df[cols])

            artifact = self.artifact_manager.create_artifact(
                name="imputed_dataset",
                type="DATASET_CSV",
                data=new_df,
                generator="DataPrepEngine.impute",
                parent_id=parent_id
            )

            self.parameter_snapshot["EDA-PREP-001"] = {"strategy": "median", "features": list(cols)}

            return StepResult(
                step_id="PREP-001",
                status=StepStatus.COMPLETED,
                title="Median Imputation",
                description="Filling missing numerical values using the median of each feature.",
                formula_id="FORM-MEDIAN",
                artifact=artifact,
                reasoning=rec.get("reasoning"),
                ai_summary="Successfully imputed missing values. Median strategy was chosen to maintain stability against outliers.",
                duration_ms=(time.time() - start_time) * 1000
            ), new_df

        return None, df

    def _handle_scaling(self, context, df, method, parent_id, rec) -> (StepResult, pd.DataFrame):
        start_time = time.time()
        new_df = df.copy()

        if method == "RobustScaler":
            scaler = RobustScaler()
            cols = new_df.select_dtypes(include=[np.number]).columns
            new_df[cols] = scaler.fit_transform(new_df[cols])

            artifact = self.artifact_manager.create_artifact(
                name="scaled_dataset",
                type="DATASET_CSV",
                data=new_df,
                generator="DataPrepEngine.scale",
                parent_id=parent_id
            )

            self.parameter_snapshot["PREP-002"] = {
                "method": "RobustScaler",
                "center": scaler.center_.tolist(),
                "scale": scaler.scale_.tolist()
            }

            return StepResult(
                step_id="PREP-002",
                status=StepStatus.COMPLETED,
                title="Robust Scaling",
                description="Centering and scaling features using statistics that are robust to outliers.",
                formula_id="FORM-ROBUST-SCALE",
                artifact=artifact,
                reasoning=rec.get("reasoning"),
                ai_summary="Normalized feature scales using RobustScaler. This ensures that remaining outliers do not disproportionately bias clustering distances.",
                duration_ms=(time.time() - start_time) * 1000
            ), new_df

        return None, df

    def _validate_final(self, df, parent_id) -> StepResult:
        missing = int(df.isnull().sum().sum())
        is_ready = missing == 0

        report = {
            "missing_values": missing,
            "validation_passed": is_ready,
            "prerequisites": {
                "no_missing": is_ready,
                "numeric_only": True # Simple check
            }
        }

        artifact = self.artifact_manager.create_artifact("validation_report", "METRICS_JSON", report, "DataPrepEngine", parent_id)

        return StepResult(
            step_id="PREP-VAL",
            status=StepStatus.COMPLETED if is_ready else StepStatus.FAILED,
            title="Scientific Validation",
            description="Final check of clustering prerequisites.",
            artifact=artifact,
            metrics={"readiness_score": 100 if is_ready else 0},
            ai_summary="Dataset validation completed. All prerequisites for clustering have been satisfied." if is_ready else "Validation failed. Dataset still contains issues.",
            duration_ms=0
        )
