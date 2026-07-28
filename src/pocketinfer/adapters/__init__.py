from pocketinfer.adapters.glm52 import Glm52Adapter
from pocketinfer.adapters.kimi_k3 import KimiK3Adapter

DEFAULT_ADAPTERS = (KimiK3Adapter(), Glm52Adapter())

__all__ = ["DEFAULT_ADAPTERS", "Glm52Adapter", "KimiK3Adapter"]
