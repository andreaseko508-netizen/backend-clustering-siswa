import os
from typing import Dict, Any, List
from sdk.models import ExecutionResult

class PublicationEngine:
    @staticmethod
    def generate_markdown_report(result: ExecutionResult, metadata: Dict[str, Any]) -> str:
        report = f"# Research Report: {metadata.get('name', 'Untitled Experiment')}\n\n"
        report += f"## Abstract\n{metadata.get('abstract', 'N/A')}\n\n"
        report += "## Methodology\n"
        report += f"- Algorithm: {metadata.get('algorithm', 'N/A')}\n"
        report += f"- Seed: {metadata.get('seed', 'N/A')}\n"
        report += f"- Environment: {metadata.get('env', 'N/A')}\n\n"

        report += "## Results\n"
        for k, v in result.metrics.items():
            report += f"- **{k}**: {v}\n"

        report += "\n## Interpretations\n"
        interpretations = result.metrics.get("interpretations", [])
        for inter in interpretations:
            report += f"### {inter.get('metric')}\n"
            report += f"- *Interpretation*: {inter.get('interpretation')}\n"
            report += f"- *Recommendation*: {inter.get('recommendation')}\n"

        return report

    @staticmethod
    def generate_latex_snippet(result: ExecutionResult) -> str:
        latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|r|}\n\\hline\n"
        latex += "Metric & Value \\\\ \\hline\n"
        for k, v in result.metrics.items():
            if isinstance(v, (int, float)):
                latex += f"{k.replace('_', ' ')} & {v:.4f} \\\\ \n"
        latex += "\\hline\n\\end{tabular}\n\\caption{Scientific Metrics}\n\\end{table}"
        return latex
