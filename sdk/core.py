from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from uuid import UUID
from enum import Enum
import time

class ExecutionMode(Enum):
    FAST = "FAST"
    RESEARCH = "RESEARCH"
    LEARNING = "LEARNING"
    DEBUG = "DEBUG"

class StepStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    WARNING = "WARNING"
    FAILED = "FAILED"

class Artifact(BaseModel):
    id: str
    name: str
    type: str
    path: str
    version: int = 1
    parent_id: Optional[str] = None
    checksum: Optional[str] = None
    generator: str
    engine_version: str = "1.0.0"
    dataset_fingerprint: Optional[str] = None
    random_seed: int = 42
    file_size_bytes: int = 0
    mime_type: str = "application/octet-stream"
    created_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = {}

class StepResult(BaseModel):
    step_id: str
    status: StepStatus
    title: str
    description: str
    formula_id: Optional[str] = None
    sample_calculation: Optional[str] = None
    artifact: Optional[Artifact] = None
    metrics: Dict[str, Any] = {}
    confidence: float = 1.0 # 0.0 to 1.0
    reasoning: Optional[str] = None
    ai_summary: Optional[str] = None
    duration_ms: float = 0.0
    memory_mb: float = 0.0
    visualization_id: Optional[str] = None

class ExecutionContext(BaseModel):
    experiment_id: UUID
    execution_id: UUID
    dataset_id: str
    random_seed: int = 42
    parameters: Dict[str, Any] = {}
    mode: ExecutionMode = ExecutionMode.RESEARCH
    artifact_path: str
    temp_path: str
    lineage_registry: Dict[str, List[str]] = {} # id -> [children]
    user_info: Dict[str, Any] = {}

class ScientificAlgorithm:
    def initialize(self, context: ExecutionContext) -> List[StepResult]:
        pass

    def preprocess(self, context: ExecutionContext) -> List[StepResult]:
        pass

    def execute(self, context: ExecutionContext) -> List[StepResult]:
        pass

    def evaluate(self, context: ExecutionContext) -> List[StepResult]:
        pass

    def visualize(self, context: ExecutionContext) -> List[StepResult]:
        pass

    def finalize(self, context: ExecutionContext) -> List[StepResult]:
        pass
