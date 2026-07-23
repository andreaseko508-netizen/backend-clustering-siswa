from abc import ABC, abstractmethod
from sdk.models import ExecutionContext, ExecutionResult

class BaseResearchPlugin(ABC):
    @abstractmethod
    def get_plugin_id(self) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        pass
