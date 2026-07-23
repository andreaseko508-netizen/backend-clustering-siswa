from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Callable

class Formula(BaseModel):
    id: str
    name: str
    latex: str
    explanation: str
    symbols: Dict[str, str] # symbol -> description
    sample_generator: Optional[Callable[[Dict[str, Any]], str]] = None
    python_logic: Optional[Callable] = None
    references: List[str] = []

class Visualization(BaseModel):
    id: str
    type: str # HISTOGRAM, SCATTER, HEATMAP, RADAR
    input_artifact_id: str
    renderer_hints: Dict[str, Any] = {}

class FormulaRegistry:
    _formulas: Dict[str, Formula] = {}

    @classmethod
    def register(cls, formula: Formula):
        cls._formulas[formula.id] = formula

    @classmethod
    def get(cls, formula_id: str) -> Optional[Formula]:
        return cls._formulas.get(formula_id)

class VisualizationRegistry:
    _visualizations: Dict[str, Visualization] = {}

    @classmethod
    def register(cls, viz: Visualization):
        cls._visualizations[viz.id] = viz

    @classmethod
    def get(cls, viz_id: str) -> Optional[Visualization]:
        return cls._visualizations.get(viz_id)
