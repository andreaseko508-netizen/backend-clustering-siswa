from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
from sdk.models import ExecutionContext, ExecutionResult

class OptimizationStrategy(ABC):
    @abstractmethod
    def suggest_parameters(self) -> List[Dict[str, Any]]:
        pass

class GridSearchStrategy(OptimizationStrategy):
    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid

    def suggest_parameters(self) -> List[Dict[str, Any]]:
        import itertools
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

class GenericOptimizer:
    def __init__(self, plugin: Any, strategy: OptimizationStrategy):
        self.plugin = plugin
        self.strategy = strategy

    def find_optimum(self, context: ExecutionContext, metric_name: str = "silhouette_score") -> ExecutionResult:
        best_result = None
        best_score = -float('inf')

        param_list = self.strategy.suggest_parameters()

        for params in param_list:
            context.parameters.update(params)
            result = self.plugin.execute(context)

            if result.status == "SUCCESS":
                score = result.metrics.get(metric_name, -float('inf'))
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_result.metrics["optimized_params"] = params

        return best_result or ExecutionResult(status="FAILED", metrics={}, artifacts=[], error_message="Optimization failed to find a valid result")
