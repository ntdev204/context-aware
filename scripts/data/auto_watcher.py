"""Auto-watcher for ROI dataset batches.

Chạy ngầm trên Laptop Windows. Theo dõi thư mục roi_dataset/,
khi phát hiện file .tar.gz mới từ Jetson sẽ tự động:
    1. Giải nén archive
    2. Chạy auto-label → phân loại ảnh vào các intent classes
    3. Dọn dẹp file tạm

Usage:
    python scripts/data/auto_watcher.py
    python scripts/data/auto_watcher.py --watch D:/nckh/context-aware/roi_dataset --output D:/nckh/context-aware/intent_dataset

Cài dependency (1 lần):
    pip install watchdog
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watcher] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WATCH_DIR = "D:/nckh/context-aware/roi_dataset"
DEFAULT_OUTPUT_DIR = "D:/nckh/context-aware/intent_dataset"
STABLE_WAIT_S = 5  # Giây chờ sau khi file ngừng tăng kích thước
# (đảm bảo rsync đã xong trước khi xử lý)


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------


def process_archive(
    archive: Path,
    output_dir: Path,
    auto_train: bool = False,
    training_cfg: dict | None = None,
) -> None:
    """Extract and auto-label one .tar.gz archive, then optionally trigger training pipeline."""
    # Import here to avoid circular import in tests
    from autolabel import extract_archives, run_autolabel

    logger.info("New batch detected: %s (%.1f MB)", archive.name, archive.stat().st_size / 1e6)

    extract_dir = archive.parent / "_extracted" / archive.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting -> %s", extract_dir)
    try:
        extract_archives(archive.parent / archive.name, extract_dir)
    except Exception as exc:
        logger.error("Extract failed: %s", exc)
        return

    logger.info("Auto-labeling -> %s", output_dir)
    try:
        stats = run_autolabel(batch_dir=extract_dir, output_dir=output_dir)
    except Exception as exc:
        logger.error("Auto-label failed: %s", exc)
        return


    # Log phân phối nhãn
    total = sum(v for k, v in stats.items() if k != "short_track_skipped")
    label_summary = "  ".join(
        f"{lbl}={stats[lbl]}({stats[lbl] / total * 100:.0f}%)" if total else f"{lbl}=0"
        for lbl in (
            "STATIONARY",
            "APPROACHING",
            "DEPARTING",
            "CROSSING",
            "FOLLOWING",
            "ERRATIC",
            "UNCERTAIN",
        )
    )
    logger.info("Labels: %s  |  skipped_short=%d", label_summary, stats["short_track_skipped"])

    # Dọn dẹp extracted dir
    try:
        shutil.rmtree(extract_dir)
    except Exception:
        pass

    # Đánh dấu archive đã xử lý (rename thay vì xóa — an toàn hơn)
    done_path = archive.with_name(f"{archive.name}.done")
    archive.rename(done_path)
    logger.info("Done: renamed to %s", done_path.name)

    if auto_train and training_cfg is not None:
        _run_orchestration_pipeline(output_dir, training_cfg)


def _run_orchestration_pipeline(dataset_dir: Path, cfg: dict) -> None:
    """Run explore -> validate -> fine-tune pipeline via subprocesses to avoid CUDA/memory leaks."""
    logger.info("[Orchestrator] ── Pipeline Started ──")

    # 1. EXPLORE
    logger.info("[Orchestrator] 1/3: Running Data Exploration...")
    explore_cmd = [
        sys.executable,
        "scripts/data/explore_roi.py",
        "--dataset",
        str(dataset_dir),
        "--output",
        str(dataset_dir),
    ]
    try:
        subprocess.run(explore_cmd, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.error("[Orchestrator] Exploration failed: %s", e)
        return

    reports_dir = dataset_dir / "reports"
    reports = sorted(reports_dir.glob("exploration_*.json"))
    if not reports:
        logger.error("[Orchestrator] No exploration report found.")
        return
    latest_report = reports[-1]

    # 2. VALIDATE
    logger.info("[Orchestrator] 2/3: Validating Dataset (%s)...", latest_report.name)
    val_cmd = [sys.executable, "scripts/data/validate_dataset.py", str(latest_report)]
    val_result = subprocess.run(val_cmd, capture_output=True, text=True)

    for line in val_result.stdout.strip().split("\n"):
        if "❌" in line or "⚠️" in line:
            logger.warning("  %s", line)
        else:
            logger.info("  %s", line)

    if val_result.returncode == 1:
        logger.warning("[Orchestrator] Pipeline BLOCKED by validation. Skipping training.")
        return

    # 3. FINE-TUNE
    epochs = cfg.get("epochs_per_finetune", 10)
    logger.info("[Orchestrator] 3/3: Fine-tuning (epochs=%d)...", epochs)

    model_dir = Path("models/cnn_intent")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "intent_v1.pt"

    train_cmd = [
        sys.executable,
        "scripts/train/train_intent_cnn.py",
        "--dataset",
        str(dataset_dir),
        "--epochs",
        str(epochs),
        "--output",
        str(model_dir),
        "--lr",
        str(cfg.get("learning_rate", 1e-4)),
        "--replay-buffer",
        str(cfg.get("replay_buffer_size", 2000)),
        "--ewc-lambda",
        str(cfg.get("ewc_lambda", 5000)),
    ]

    if model_path.exists():
        logger.info("  [Hint] Found existing checkpoint -> Continual Training.")
        train_cmd.extend(["--resume", str(model_path), "--epochs-are-additional"])

    try:
        # Stream output directly to terminal so user can see progress bar
        subprocess.run(train_cmd, check=True)
        logger.info("[Orchestrator] ✅ Pipeline Complete! Model saved -> %s", model_path)
    except subprocess.CalledProcessError as e:
        logger.error("[Orchestrator] ❌ Training failed: %s", e)


def is_file_stable(path: Path, wait: int = STABLE_WAIT_S) -> bool:
    """Return True if file size is unchanged for `wait` seconds."""
    try:
        size1 = path.stat().st_size
        time.sleep(wait)
        size2 = path.stat().st_size
        return size1 == size2 and size1 > 0
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Watcher  (dùng watchdog nếu có, fallback polling nếu không)
# ---------------------------------------------------------------------------


def run_with_watchdog(
    watch_dir: Path,
    output_dir: Path,
    auto_train: bool,
    training_cfg: dict,
) -> None:
    from watchdog.events import FileCreatedEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    class BatchHandler(FileSystemEventHandler):
        def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix == ".zip" or path.name.endswith(".tar.gz"):
                logger.info("FileCreated event: %s", path.name)
                if is_file_stable(path):
                    process_archive(path, output_dir, auto_train, training_cfg)

    observer = Observer()
    observer.schedule(BatchHandler(), str(watch_dir), recursive=False)
    observer.start()
    logger.info("Watchdog observer started on: %s", watch_dir)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def run_polling(
    watch_dir: Path,
    output_dir: Path,
    auto_train: bool,
    training_cfg: dict,
    poll_interval: int = 10,
) -> None:
    """Fallback polling loop (no watchdog dependency)."""
    seen: set[Path] = set()

    # Seed with pre-existing files so they are not re-processed
    seen.update(watch_dir.glob("*.tar.gz"))
    seen.update(watch_dir.glob("*.zip"))
    seen.update(watch_dir.glob("*.done"))

    logger.info(
        "Polling mode — checking every %ds for new batches in: %s", poll_interval, watch_dir
    )

    try:
        while True:
            archives = list(watch_dir.glob("*.tar.gz")) + list(watch_dir.glob("*.zip"))
            for archive in sorted(archives):
                if archive not in seen:
                    seen.add(archive)
                    logger.info("Polling detected: %s", archive.name)
                    if is_file_stable(archive):
                        process_archive(archive, output_dir, auto_train, training_cfg)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("Watcher stopped by user.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-watcher: monitors roi_dataset/ and labels batches automatically.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--watch",
        type=Path,
        default=DEFAULT_WATCH_DIR,
        help="Directory to watch for .zip or .tar.gz batches",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output labeled dataset directory"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=30,
        help="[Deprecated — no longer used] Kept for backwards CLI compatibility only.",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=15,
        help="[Deprecated — no longer used] Kept for backwards CLI compatibility only.",
    )
    parser.add_argument("--poll", action="store_true", help="Force polling mode (skip watchdog)")
    parser.add_argument(
        "--interval", type=int, default=10, help="Poll interval in seconds (polling mode only)"
    )
    args = parser.parse_args()

    watch_dir = args.watch
    output_dir = args.output

    watch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add scripts/ to path so autolabel.py can be imported
    scripts_dir = Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    logger.info("ROI Auto-Watcher | watch=%s", watch_dir)
    logger.info("  Output: %s", output_dir)
    logger.info("  Labeling thresholds managed internally by autolabel.py")

    # Load training config
    training_yaml = Path("config/training.yaml")
    training_cfg = {}
    auto_train = False

    if training_yaml.exists():
        try:
            with open(training_yaml) as f:
                t_data = yaml.safe_load(f)
                if t_data and "training" in t_data:
                    training_cfg = t_data["training"]
                    auto_train = training_cfg.get("auto_train", False)
        except Exception as e:
            logger.error("Failed to load %s: %s", training_yaml, e)

    if auto_train:
        logger.info("  Orchestrator: ENABLED (Explore -> Validate -> Train)")
    else:
        logger.info("  Orchestrator: DISABLED (Extract & Label only)")

    logger.info("---")

    use_watchdog = False
    if not args.poll:
        try:
            import watchdog  # noqa: F401

            use_watchdog = True
            logger.info("Using watchdog (event-driven, low CPU)")
        except ImportError:
            logger.warning(
                "watchdog not installed — falling back to polling. Install with: pip install watchdog"
            )

    if use_watchdog:
        run_with_watchdog(watch_dir, output_dir, auto_train, training_cfg)
    else:
        run_polling(watch_dir, output_dir, auto_train, training_cfg, args.interval)


if __name__ == "__main__":
    main()
