from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "config"


class Config:

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val: Any = self._data
        for k in keys:
            if not isinstance(val, dict):
                return default
            val = val.get(k, default)
        return val

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def section(self, key: str) -> Config:
        val = self.get(key, {})
        return Config(val if isinstance(val, dict) else {})

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        env_path = os.environ.get("CONFIG_PATH")
        if env_path:
            path = Path(env_path)
        else:
            mode = os.environ.get("MODE", "development")
            path = _DEFAULT_CONFIG_DIR / f"{mode}.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        data = yaml.safe_load(f) or {}

    logger.info("Config loaded from %s", path)
    return Config(data)


def setup_logging(cfg: Config) -> None:
    from .logging_config import setup_logging as _setup

    _setup(cfg)
