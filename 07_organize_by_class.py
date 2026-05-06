"""
Organize Dataset by Class
=========================
Creates two folder structures:
  - "With Helmet"    -> images/labels where class 0 (helmet) is present
  - "Without Helmet" -> images/labels where class 1 (no_helmet) is present
                        OR background images (empty label files)

Each folder contains: train / valid / test subfolders with images + labels.

Source: D:\Helmet_detection\Dataset  (original un-filtered labels)
Output: D:\Helmet_detection\With Helmet
        D:\Helmet_detection\Without Helmet
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path(r"D:\Helmet_detection")

# Original source datasets (labels still have both classes)
SOURCE_DIRS = [
    BASE_DIR / "Dataset" / "two wheeler helmet.v1i.yolov8",
    BASE_DIR / "Dataset" / "two wheeler helmet.v2i.yolov8",
    BASE_DIR / "Dataset" / "two wheeler helmet.v3i.yolov8",
    BASE_DIR / "Dataset" / "two wheeler helmet.v4i.yolov8",
    BASE_DIR / "Dataset" / "two wheeler helmet.v5i.yolov8",
    BASE_DIR / "Dataset" / "Helmet_Dataset",
]

# Output directories
WITH_HELMET_DIR    = BASE_DIR / "With Helmet"
WITHOUT_HELMET_DIR = BASE_DIR / "Without Helmet"

SPLITS = {
    "train": "train",
    "valid": "valid",
    "test":  "test",
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def make_structure():
    """Create all output subdirectories."""
    for base in [WITH_HELMET_DIR, WITHOUT_HELMET_DIR]:
        for split in SPLITS.values():
            (base / split / "images").mkdir(parents=True, exist_ok=True)
            (base / split / "labels").mkdir(parents=True, exist_ok=True)
    print("Output folder structure created.")


def classify_label(label_path: Path):
    """
    Read a YOLO label file and return which classes are present.
    Returns a set of class IDs (e.g. {0}, {1}, {0,1}, or {} for background).
    """
    if not label_path.exists():
        return set()
    classes = set()
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(int(parts[0]))
    return classes


def copy_file(src: Path, dest_dir: Path):
    """Copy a file to dest_dir, creating it if needed."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)


def process_source(source_dir: Path, split_key: str, split_out: str, stats: dict):
    """
    Process one split (train/valid/test) from one source dataset.
    Copies images + labels to With Helmet / Without Helmet accordingly.
    """
    # Try both 'valid' and 'val' folder naming conventions
    img_dir = source_dir / split_key / "images"
    lbl_dir = source_dir / split_key / "labels"

    if not img_dir.exists():
        return  # This split doesn't exist in this source dataset

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        classes_present = classify_label(lbl_path)

        # Determine which folder(s) the image belongs to
        placed = False

        if 0 in classes_present:
            # Has helmet annotations
            copy_file(img_path, WITH_HELMET_DIR / split_out / "images")
            if lbl_path.exists():
                copy_file(lbl_path, WITH_HELMET_DIR / split_out / "labels")
            stats['with_helmet'] += 1
            placed = True

        if 1 in classes_present:
            # Has no_helmet annotations
            copy_file(img_path, WITHOUT_HELMET_DIR / split_out / "images")
            if lbl_path.exists():
                copy_file(lbl_path, WITHOUT_HELMET_DIR / split_out / "labels")
            stats['without_helmet'] += 1
            placed = True

        if not placed:
            # Background image (no annotations) → Without Helmet
            copy_file(img_path, WITHOUT_HELMET_DIR / split_out / "images")
            if lbl_path.exists():
                copy_file(lbl_path, WITHOUT_HELMET_DIR / split_out / "labels")
            stats['background'] += 1


def main():
    print("=" * 60)
    print("ORGANIZING DATASET BY CLASS")
    print("=" * 60)
    print(f"\nOutput: {WITH_HELMET_DIR}")
    print(f"Output: {WITHOUT_HELMET_DIR}\n")

    make_structure()

    stats = {'with_helmet': 0, 'without_helmet': 0, 'background': 0}

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            print(f"  [SKIP] Not found: {source_dir.name}")
            continue
        print(f"  Processing: {source_dir.name}")

        # Handle both 'valid' and 'val' naming
        for split_key, split_out in SPLITS.items():
            process_source(source_dir, split_key, split_out, stats)
            if split_key == "valid":
                # Some datasets use 'val' instead of 'valid'
                process_source(source_dir, "val", split_out, stats)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\n  Images copied to 'With Helmet'    : {stats['with_helmet']}")
    print(f"  Images copied to 'Without Helmet' : {stats['without_helmet']}")
    print(f"  Background images (no annotation) : {stats['background']}")
    print(f"\n  With Helmet    -> {WITH_HELMET_DIR}")
    print(f"  Without Helmet -> {WITHOUT_HELMET_DIR}")


if __name__ == '__main__':
    main()
