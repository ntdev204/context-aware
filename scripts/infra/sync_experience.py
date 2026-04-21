"""Dataset and Experience sync daemon (run on Jetson host via Docker).

Periodically rsyncs HDF5 logs AND JPG ROI datasets to a remote target (laptop) over SSH.
Uses --remove-source-files to ensure Jetson storage doesn't run out.

Usage: python scripts/infra/sync_experience.py [--once] [--interval S]
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("sync_experience")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def rsync_to_target(log_dir: Path, target: str, ssh_key: str | None) -> None:
    """Push files to remote target using rsync over SSH.

    Uses directory sync with --remove-source-files to automatically
    manage local storage (Jetson deletes after sending to laptop).
    """
    ssh_cmd = "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"
    if ssh_key:
        ssh_cmd += f" -i {ssh_key}"

    cmd = [
        "rsync",
        "--archive",
        "--compress",
        "--human-readable",
        "--remove-source-files",
        "--include=*.h5",
        "--include=*.jpg",
        "--exclude=*",
        "-e", ssh_cmd,
        f"{log_dir}/",
        f"{target}/",
    ]

    # Quick check if there's anything to sync
    has_files = any(_ for _ in log_dir.glob("*.h5")) or any(_ for _ in log_dir.glob("*.jpg"))
    if not has_files:
        return

    logger.info("Triggering directory sync %s → %s", log_dir, target)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info("rsync complete. Local files safely removed.")
        else:
            logger.error("rsync failed (rc=%d):\n%s", result.returncode, result.stderr.strip()[:500])
    except subprocess.TimeoutExpired:
        logger.error("rsync timed out after 300s — check network.")
    except FileNotFoundError:
        logger.error("rsync binary not found. Install: apt-get install rsync")


def run_daemon(log_dir: Path, target: str, interval: float, ssh_key: str | None) -> None:
    """Run sync loop indefinitely."""
    logger.info("Sync daemon started | dir=%s | target=%s | interval=%.0fs", log_dir, target, interval)
    while True:
        try:
            rsync_to_target(log_dir, target, ssh_key)
        except Exception as exc:
            logger.error("Sync cycle error: %s", exc, exc_info=True)

        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync HDF5/JPG files from Jetson to server.")
    parser.add_argument("--log-dir", default=os.getenv("SYNC_LOG_DIR", "logs/experience"), help="Local sync dir")
    parser.add_argument("--target", default=os.getenv("SYNC_TARGET"), help="rsync destination")
    parser.add_argument("--interval", type=float, default=float(os.getenv("SYNC_INTERVAL", "30")))
    parser.add_argument("--ssh-key", default=os.getenv("SYNC_SSH_KEY", str(Path.home() / ".ssh" / "sync_key")))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    _setup_logging(args.log_level)

    if not args.target:
        logger.error("SYNC_TARGET is required.")
        sys.exit(1)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ssh_key = args.ssh_key if Path(args.ssh_key).exists() else None

    if args.once:
        rsync_to_target(log_dir, args.target, ssh_key)
    else:
        run_daemon(log_dir, args.target, args.interval, ssh_key)

if __name__ == "__main__":
    main()
