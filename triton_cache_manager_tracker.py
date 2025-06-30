from typing import Dict, Optional, Any
from triton.runtime.cache import CacheManager, FileCacheManager
from triton import knobs
from collections import Counter, defaultdict
from threading import Lock


class GlobalCacheTrackerStats:
    """Aggregate cache hit / miss statistics for every Triton cache lookup."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._totals: Counter = Counter()
        self._per_key: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"name": "<unknown>", "hits": 0, "misses": 0}
        )

    def update(self, cache_key: str, kernel_name: str, hit: bool) -> None:
        """Record one cache lookup."""
        with self._lock:
            self._totals["access"] += 1
            self._totals["hit" if hit else "miss"] += 1

            entry = self._per_key[cache_key]
            entry["name"] = kernel_name
            if hit:
                entry["hits"] += 1
            else:
                entry["misses"] += 1

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep-copy of current stats."""
        with self._lock:
            return {
                "totals": self._totals.copy(),
                "per_key": {k: v.copy() for k, v in self._per_key.items()},
            }

    def reset(self) -> None:
        """Reset everything"""
        with self._lock:
            self._totals.clear()
            self._per_key.clear()


# Instantiate the global tracker
_tracker = GlobalCacheTrackerStats()


class TrackingCacheManager(CacheManager):
    def __init__(self, key: str, override: bool = False, dump: bool = False):
        self.full_cache_key = key

        # Determine the base cache manager to wrap.
        # It also allows wrapping another custom manager
        base_cache_cls = FileCacheManager
        if knobs.cache.manager_class is not TrackingCacheManager:
            base_cache_cls = knobs.cache.manager_class

        self._base_manager = base_cache_cls(key, override=override, dump=dump)

    def get_file(self, filename: str) -> Optional[str]:
        """Intercepts file retrieval to track hits/misses."""
        return self._base_manager.get_file(filename)

    def put(self, data, filename: str, binary: bool = True) -> str:
        """Passes through put operations to the base manager."""
        return self._base_manager.put(data, filename, binary)

    def get_group(self, filename: str):
        group = self._base_manager.get_group(filename)
        kernel_name = filename.split(".")[0]

        _tracker.update(
            cache_key=self.full_cache_key,
            kernel_name=kernel_name,
            hit=group is not None,
        )
        return group

    def put_group(self, filename: str, group: Dict[str, str]):
        """Passes through put_group operations to the base manager."""
        return self._base_manager.put_group(filename, group)
