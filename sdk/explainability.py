from typing import Dict, Any, List

class PrescriptiveExplainabilityEngine:
    @staticmethod
    def interpret_metrics(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        interpretations = []

        # Silhouette Score Interpretation
        if "silhouette_score" in metrics:
            val = metrics["silhouette_score"]
            if val > 0.7:
                text = "Strong structure found. Clusters are well-separated and clearly defined."
                advice = "Excellent. No further action needed unless feature importance is required."
            elif val > 0.5:
                text = "Reasonable structure. Some overlap exists but clusters are distinct."
                advice = "Consider minor outlier removal or slight adjustment of K."
            elif val > 0.25:
                text = "Weak structure. Clusters are likely overlapping significantly."
                advice = "Try feature engineering or a density-based algorithm like DBSCAN."
            else:
                text = "No substantial structure. Data points are not forming coherent clusters."
                advice = "Major data cleaning required. Check for extreme outliers or irrelevant features."

            interpretations.append({
                "metric": "silhouette_score",
                "value": str(val),
                "interpretation": text,
                "recommendation": advice
            })

        # Davies-Bouldin Index Interpretation
        if "davies_bouldin_index" in metrics:
            val = metrics["davies_bouldin_index"]
            if val < 0.5:
                text = "Excellent compactness and separation."
                advice = "Model is highly stable."
            elif val < 1.0:
                text = "Good balance between cluster density and separation."
                advice = "Valid clustering result."
            else:
                text = "High cluster overlap or dispersion detected."
                advice = "Try increasing the number of clusters or using PCA to reduce noise."

            interpretations.append({
                "metric": "davies_bouldin_index",
                "value": str(val),
                "interpretation": text,
                "recommendation": advice
            })

        return interpretations
