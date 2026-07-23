import hashlib
import os
import json
import time
from typing import Any, Dict, List, Optional
from sdk.core import Artifact

class ArtifactManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.registry_path = os.path.join(base_path, "artifact_registry.json")
        self._registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self._registry, f, indent=2)

    def create_artifact(self, name: str, type: str, data: Any,
                        generator: str, parent_id: Optional[str] = None,
                        metadata: Dict[str, Any] = {}) -> Artifact:

        artifact_id = f"ART-{int(time.time() * 1000)}"
        file_ext = self._get_extension(type)
        file_name = f"{artifact_id}_{name}.{file_ext}"
        file_path = os.path.join(self.base_path, file_name)

        # Save data
        self._save_data(file_path, type, data)

        # Calculate checksum
        checksum = self._calculate_sha256(file_path)

        artifact = Artifact(
            id=artifact_id,
            name=name,
            type=type,
            path=file_path,
            parent_id=parent_id,
            checksum=checksum,
            generator=generator,
            metadata=metadata
        )

        self._registry[artifact_id] = artifact.dict()
        self._save_registry()
        return artifact

    def _get_extension(self, type: str) -> str:
        mapping = {
            "DATASET_CSV": "csv",
            "MATRIX_NPY": "npy",
            "METRICS_JSON": "json",
            "MODEL_PKL": "pkl"
        }
        return mapping.get(type, "dat")

    def _save_data(self, path: str, type: str, data: Any):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if type == "DATASET_CSV":
            data.to_csv(path, index=False)
        elif type == "METRICS_JSON":
            with open(path, 'w') as f:
                json.dump(data, f)
        # Add other types as needed

    def _calculate_sha256(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_lineage(self, artifact_id: str) -> List[str]:
        lineage = []
        current = artifact_id
        while current and current in self._registry:
            lineage.append(current)
            current = self._registry[current].get("parent_id")
        return lineage
