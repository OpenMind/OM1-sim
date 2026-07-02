"""Shared helpers for the per-robot policies (no Isaac Sim dependency)."""

import numpy as np
import yaml


def _load_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except Exception:
        return {}


def _expand_param(value, size: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(size, default, dtype=np.float32)
    if isinstance(value, (int, float)):
        return np.full(size, float(value), dtype=np.float32)
    arr = np.array(value, dtype=np.float32)
    if arr.size != size:
        return np.full(size, default, dtype=np.float32)
    return arr
