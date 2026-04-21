"""HDF5 experience sync daemon (run on Jetson host, not inside Docker).

Periodically caps local .h5 files to SYNC_CAP (default 5000) and rsyncs
stable files to a remote target (laptop/server) over SSH.

Usage: python scripts/sync_experience.py [--once] [--interval S] [--cap N]
See: docs/ for full SSH key setup instructions.
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


def _get_h5_files(log_dir: Path) -> list[Path]:
    """Return all .h5 files sorted by mtime ascending (oldest first)."""
    files = list(log_dir.glob("*.h5"))
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def _split_by_stability(files: list[Path], stable_age_s: float) -> tuple[list[Path], list[Path]]:
    """Split files into (stable, active) based on time since last modification."""
    now = time.time()
    stable, active = [], []
    for f in files:
        try:
            age = now - f.stat().st_mtime
            (stable if age >= stable_age_s else active).append(f)
        except OSError:
            pass  # file disappeared between glob and stat
    return stable, active


def enforce_cap(log_dir: Path, cap: int, stable_age_s: float) -> list[Path]:
    """Enforce local file cap. Only deletes stable files (oldest first).

    Returns the list of stable files remaining after cap enforcement.
    Active (recently written) files are never deleted.
    """
    all_files = _get_h5_files(log_dir)
    stable, active = _split_by_stability(all_files, stable_age_s)
    total = len(all_files)

    if total > cap:
        excess = total - cap
        # Delete from stable oldest first; never touch active files
        to_delete = stable[:excess]
        for f in to_delete:
            try:
                f.unlink()
                logger.info("Cap (%d): removed %s", cap, f.name)
            except OSError as err:
                logger.warning("Could not remove %s: %s", f.name, err)
        stable = stable[excess:]
        logger.info(
            "Cap enforced: deleted %d file(s), local total now %d",
            len(to_delete),
            len(stable) + len(active),
        )

    _ = active  # active files are never pruned
    return stable


def rsync_to_target(stable_files: list[Path], target: str, ssh_key: str | None) -> None:
    """Push stable files to remote target using rsync over SSH.

    --ignore-existing: never overwrite a file that already exists on the remote.
    This means the remote (laptop) accumulates everything; Jetson only sends new files.
    """
    if not stable_files:
        logger.debug("No stable files to sync.")
        return

    ssh_cmd = "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"
    if ssh_key:
        ssh_cmd += f" -i {ssh_key}"

    cmd = [
        "rsync",
        "--archive",           # preserve timestamps / permissions
        "--ignore-existing",   # remote side keeps all; never re-send
        "--compress",          # gzip in transit (h5 files compress well)
        "--human-readable",
        "-e", ssh_cmd,
    ] + [str(f) for f in stable_files] + [f"{target}/"]

    logger.info("Syncing %d stable file(s) → %s", len(stable_files), target)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info("Sync complete. %d file(s) sent.", len(stable_files))
            if result.stdout.strip():
                logger.debug("rsync output:\n%s", result.stdout.strip())
        else:
            logger.error("rsync failed (rc=%d):\n%s", result.returncode, result.stderr.strip())
    except subprocess.TimeoutExpired:
        logger.error("rsync timed out after 300s — check network connectivity.")
    except FileNotFoundError:
        logger.error("rsync binary not found. Install: apt-get install rsync")


def run_once(log_dir: Path, target: str, cap: int, stable_age_s: float, ssh_key: str | None) -> None:
    """Run a single cap+sync cycle."""
    stable = enforce_cap(log_dir, cap, stable_age_s)
    rsync_to_target(stable, target, ssh_key)


def run_daemon(
    log_dir: Path,
    target: str,
    cap: int,
    interval: float,
    stable_age_s: float,
    ssh_key: str | None,
) -> None:
    """Run sync loop indefinitely. Ctrl-C to stop."""
    logger.info(
        "Sync daemon started | dir=%s | target=%s | cap=%d | interval=%.0fs | stable_age=%.0fs",
        log_dir, target, cap, interval, stable_age_s,
    )
    while True:
        try:
            run_once(log_dir, target, cap, stable_age_s, ssh_key)
        except Exception as exc:
            logger.error("Sync cycle error: %s", exc, exc_info=True)
        logger.debug("Sleeping %.0fs until next sync...", interval)
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync HDF5 experience files from Jetson to laptop/server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-dir", default=os.getenv("SYNC_LOG_DIR", "logs/experience"),
        metavar="PATH", help="Local HDF5 directory (default: logs/experience)",
    )
    parser.add_argument(
        "--target", default=os.getenv("SYNC_TARGET"),
        metavar="USER@HOST:PATH", help="rsync destination [env: SYNC_TARGET]",
    )
    parser.add_argument(
        "--cap", type=int, default=int(os.getenv("SYNC_CAP", "5000")),
        metavar="N", help="Max local .h5 files to keep on Jetson (default: 5000)",
    )
    parser.add_argument(
        "--interval", type=float, default=float(os.getenv("SYNC_INTERVAL", "30")),
        metavar="S", help="Seconds between sync cycles (default: 30)",
    )
    parser.add_argument(
        "--stable-age", type=float, default=float(os.getenv("SYNC_STABLE_AGE", "60")),
        metavar="S", help="Seconds since last write before file is considered stable (default: 60)",
    )
    parser.add_argument(
        "--ssh-key", default=os.getenv("SYNC_SSH_KEY", str(Path.home() / ".ssh" / "sync_key")),
        metavar="PATH", help="SSH private key path [env: SYNC_SSH_KEY]",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single sync cycle then exit (for cron / manual use)",
    )
    parser.add_argument("--log-level", default="INFO", metavar="LEVEL")

    args = parser.parse_args()
    _setup_logging(args.log_level)

    # Validate
    if not args.target:
        logger.error(
            "SYNC_TARGET is required.\n"
            "  Example: SYNC_TARGET=user@192.168.1.x:/d/nckh/context-aware/logs/experience"
        )
        sys.exit(1)

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        logger.error("Log directory does not exist: %s", log_dir)
        sys.exit(1)

    ssh_key = args.ssh_key if Path(args.ssh_key).exists() else None
    if not ssh_key:
        logger.warning(
            "SSH key not found at %s. Falling back to password auth (may prompt).",
            args.ssh_key,
        )

    if args.once:
        run_once(log_dir, args.target, args.cap, args.stable_age, ssh_key)
    else:
        run_daemon(log_dir, args.target, args.cap, args.interval, args.stable_age, ssh_key)


if __name__ == "__main__":
    main()
