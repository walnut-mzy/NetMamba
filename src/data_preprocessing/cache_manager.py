"""Cache management for expensive preprocessing steps."""
from __future__ import annotations

import os
import pickle
from typing import Any, Optional

from .utils import compute_md5, get_logger


class CacheManager:
    """Save and load cached preprocessing artifacts keyed by configuration."""

    def __init__(self, cache_dir: str = "./cache", logger=None):
        self.cache_dir = cache_dir
        self.logger = logger or get_logger()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        digest = compute_md5(key)
        return os.path.join(self.cache_dir, f"{digest}.pkl")

    def load(self, key: str) -> Optional[Any]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        self.logger.info(f"Loading cached data from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, key: str, obj: Any) -> str:
        path = self._cache_path(key)
        self.logger.info(f"Saving cache to {path}")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return path


__all__ = ["CacheManager"]
