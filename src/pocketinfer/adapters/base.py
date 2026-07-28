from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from pocketinfer.models import Candidate, FidelityPolicy


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def supports(self, config: dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def candidates(
        self,
        config: dict[str, Any],
        policy: FidelityPolicy,
    ) -> Iterable[Candidate]:
        pass
