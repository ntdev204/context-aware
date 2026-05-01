import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
try:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.perception.intent_labels import canonical_label, is_trainable_label
except Exception:  # pragma: no cover
    def canonical_label(label: str | None) -> str:
        label_up = str(label or "UNCERTAIN").strip().upper()
        return "UNCERTAIN" if label_up in {"FOLLOW", "FOLLOWING"} else label_up

    def is_trainable_label(label: str | None) -> bool:
        return canonical_label(label) in {
            "STATIONARY",
            "APPROACHING",
            "DEPARTING",
            "CROSSING",
            "ERRATIC",
        }


def _append_jsonl_line(path: Path, row: dict) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    start = time.time()
    while lock_path.exists():
        if time.time() - start > 10.0:
            raise TimeoutError(f"Timed out waiting for metadata lock: {lock_path}")
        time.sleep(0.05)
    lock_path.write_text(str(time.time()), encoding="utf-8")
    try:
        with open(path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def import_dataset(export_dir: Path, output_dir: Path) -> None:
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    classes_file = labels_dir / "classes.txt"

    if not images_dir.exists() or not labels_dir.exists():
        print("[-] Invalid export format. Expected 'images/' and 'labels/' subdirectories.")
        return

    if not classes_file.exists():
        print("[-] 'classes.txt' not found in labels directory. Cannot map class IDs.")
        return

    with open(classes_file, encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    print(f"[*] Found {len(classes)} classes: {classes}")

    images = list(images_dir.glob("*.jpg"))
    print(f"[*] Found {len(images)} images to import.")

    output_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    for img_path in images:
        txt_path = labels_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():
            continue

        with open(txt_path, encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            continue

        first_line = lines[0].strip()
        if not first_line:
            continue

        parts = first_line.split()
        class_id = int(parts[0])

        if class_id >= len(classes):
            print(f"[!] Invalid class_id {class_id} in {txt_path.name}")
            continue

        class_name = canonical_label(classes[class_id])
        if not is_trainable_label(class_name):
            print(f"[!] Skipping non-trainable label {classes[class_id]} in {txt_path.name}")
            continue

        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)

        dest_path = class_dir / f"human_{img_path.name}"
        shutil.copy2(img_path, dest_path)
        _append_jsonl_line(
            output_dir / "imported_metadata.jsonl",
            {
                "file": f"{class_name}/{dest_path.name}",
                "source_file": img_path.name,
                "label": class_name,
                "label_source": "human",
                "review_status": "human_verified",
                "review_required": False,
                "ts": int(time.time() * 1000),
            },
        )
        imported += 1

    print(f"[+] Successfully imported {imported} human-labeled images to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Human Label Importer")
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to YOLO formatted export directory"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("intent_dataset/human"), help="Output dataset directory"
    )
    args = parser.parse_args()

    import_dataset(args.input, args.output)
