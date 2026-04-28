import argparse
import shutil
from pathlib import Path


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

        class_name = classes[class_id]

        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)

        dest_path = class_dir / f"human_{img_path.name}"
        shutil.copy2(img_path, dest_path)
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
