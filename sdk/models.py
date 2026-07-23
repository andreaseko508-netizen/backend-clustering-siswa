from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from uuid import UUID

class ExecutionContext(BaseModel):
    execution_id: UUID
    workflow_id: UUID
    project_id: UUID
    institution_id: UUID
    parameters: Dict[str, Any]
    input_datasets: Dict[str, str] # name -> file_path
    artifact_path: str
    temp_path: str

class ResearchArtifact(BaseModel):
    name: str
    type: str
    file_path: str
    metadata: Optional[Dict[str, Any]] = None

class ExecutionResult(BaseModel):
    status: str # SUCCESS, FAILED
    metrics: Dict[str, Any]
    artifacts: List[ResearchArtifact]
    error_message: Optional[str] = None
