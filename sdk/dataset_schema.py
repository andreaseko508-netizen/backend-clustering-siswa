import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sdk.models import ExecutionContext

class DatasetSchemaRegistry:
    @staticmethod
    def extract_schema(df: pd.DataFrame) -> Dict[str, Any]:
        schema = {
            "columns": [],
            "row_count": len(df),
            "column_count": len(df.columns)
        }
        for col in df.columns:
            col_info = {
                "name": col,
                "type": str(df[col].dtype),
                "nullable": df[col].isnull().any(),
                "unique_values": int(df[col].nunique()) if df[col].nunique() < 100 else -1
            }
            schema["columns"].append(col_info)
        return schema

    @staticmethod
    def calculate_stats(df: pd.DataFrame) -> Dict[str, Any]:
        numeric_df = df.select_dtypes(include=[np.number])
        stats = {}
        for col in numeric_df.columns:
            stats[col] = {
                "mean": float(numeric_df[col].mean()),
                "median": float(numeric_df[col].median()),
                "std": float(numeric_df[col].std()),
                "min": float(numeric_df[col].min()),
                "max": float(numeric_df[col].max()),
                "null_count": int(df[col].isnull().sum())
            }
        return stats
