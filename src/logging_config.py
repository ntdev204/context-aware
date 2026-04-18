"""Structured logging setup for the Context-Aware AI Server.

Log routing:
  src.perception.*    -> logs/perception.log   (INFO+)
  src.navigation.*    -> logs/navigation.log   (INFO+)
  src.experience.*    -> logs/experience.log   (INFO+)
  src.communication.* -> logs/communication.log (INFO+)
  src.main            -> logs/server.log        (INFO+)
  All loggers         -> logs/errors.log        (ERROR+)
  Console             -> stderr                 (ERROR+ only, plus lifecycle events)

In development mode all output goes to console at DEBUG level (no file splitting).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

# Loggers that produce high-frequency noise — suppressed to WARNING everywhere.
_NOISY_LOGGERS = (
    "ultralytics",
    "PIL",
    "h5py",
    "uvicorn.access",
    "uvicorn.error",
    "fastapi",
    "httpx",
    "asyncio",
)

_LOG_FORMAT      = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_FORMAT_CONS = "%(asctime)s [%(levelname)-8s] %(message)s"
_DATE_FORMAT     = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES       = 10 * 1024 * 1024   # 10 MB
_BACKUP_COUNT    = 5

_LEVEL_COLORS = {
    "DEBUG":    "\033[36m",     # cyan
    "INFO":     "\033[32m",     # green
    "WARNING":  "\033[33m",     # yellow
    "ERROR":    "\033[31m",     # red
    "CRITICAL": "\033[1;31m",   # bold red
}
_RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """ANSI-colored console formatter. Falls back to plain text if not a TTY."""

    def __init__(self, fmt: str, datefmt: str) -> None:
        super().__init__(fmt, datefmt=datefmt)
        self._use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not self._use_color:
            return msg
        color = _LEVEL_COLORS.get(record.levelname, "")
        return f"{color}{msg}{_RESET}"


# Component → logger name prefix mapping
_COMPONENT_LOGGERS: dict[str, str] = {
    "perception":    "src.perception",
    "navigation":    "src.navigation",
    "experience":    "src.experience",
    "communication": "src.communication",
    "server":        "src.main",
}


def _rotating_handler(path: Path, level: int) -> logging.handlers.RotatingFileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


class _LifecycleFilter(logging.Filter):
    """Allow ERROR+ OR specific lifecycle keywords through the console handler."""

    _KEYWORDS = frozenset({
        "starting", "started", "stopped", "stopping",
        "shutdown", "loaded", "ready", "exiting",
    })

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        msg = record.getMessage().lower()
        return any(kw in msg for kw in self._KEYWORDS)


def setup_logging(cfg) -> None:
    """Configure structured logging from config.

    Production: per-component rotating files + errors.log; console shows ERROR+
    and lifecycle events only.
    Development: single console handler at DEBUG level.
    """
    mode = cfg.get("system.mode", "development")
    log_dir = Path(cfg.get("logging.dir", "logs"))

    # Silence noisy third-party loggers globally
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
        logging.getLogger(name).propagate = False

    if mode != "production":
        _setup_development(cfg)
        return

    _setup_production(log_dir, cfg)


def _setup_development(cfg) -> None:
    level_str = cfg.get("system.log_level", "DEBUG")
    level = getattr(logging, level_str.upper(), logging.DEBUG)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(ColoredFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)


def get_log_dir(cfg) -> Path:
    """Return the resolved log directory path from config."""
    return Path(cfg.get("logging.dir", "logs"))


def _setup_production(log_dir: Path, cfg) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger: let handlers decide what gets written
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # Console: INFO+ with color; ERROR+ always visible.
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(ColoredFormatter(_LOG_FORMAT_CONS, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # Global error log (catches ERROR+ from ALL components)
    error_handler = _rotating_handler(log_dir / "errors.log", logging.ERROR)
    root.addHandler(error_handler)

    # Per-component file handlers with optional level overrides from YAML
    component_files = {
        "perception":    log_dir / "perception.log",
        "navigation":    log_dir / "navigation.log",
        "experience":    log_dir / "experience.log",
        "communication": log_dir / "communication.log",
        "server":        log_dir / "server.log",
    }

    for component, log_path in component_files.items():
        logger_name = _COMPONENT_LOGGERS[component]

        level_str = cfg.get(f"logging.levels.{component}", "INFO")
        level = getattr(logging, level_str.upper(), logging.INFO)

        comp_logger = logging.getLogger(logger_name)
        comp_logger.setLevel(level)
        comp_logger.propagate = True   # still reaches root (errors.log + console)
        comp_logger.addHandler(_rotating_handler(log_path, level))

    # High-frequency safety debug messages: only let WARNING+ through
    logging.getLogger("src.navigation.safety_monitor").setLevel(logging.WARNING)

    # Suppress API / streaming access logs
    for name in ("src.api", "src.streaming"):
        logging.getLogger(name).setLevel(logging.WARNING)

