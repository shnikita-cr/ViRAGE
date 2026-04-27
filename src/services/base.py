from abc import ABC, abstractmethod
from typing import Any

from src.infrastructure.runtime import RuntimeContext


class BaseService(ABC):
    @abstractmethod
    def invoke(self, *args: Any, runtime: RuntimeContext, **kwargs: Any) -> Any:
        raise NotImplementedError
