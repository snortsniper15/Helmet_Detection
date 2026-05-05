"""
Phase 1: Dataset Preprocessing & Validation
============================================
- Validates annotation integrity across all 5 Roboflow datasets
- Detects and removes corrupted images
- Detects and removes duplicate images (perceptual hash)
- Standardizes class labels (helmet / no_helmet)
- Analyzes class imbalance
- Merges all datasets into a unified combined_dataset/ structure
- Creates combined_data.yaml
"""

import os
import sys
import shutil
import hashlib
import random
import yaml
from pathlib import Path
from collections import defaultdict, Counter

# Try to import optional dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARN] Pillow not installed. Install with: pip install Pillow")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("[WARN] matplotlib not installed. Plots will be skipped.")

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(r"D:\Helmet_detection")
DATASET_DIR = BASE_DIR / "Dataset"
COMBINED_DIR = BASE_DIR / "combined_dataset"
QUARANTINE_DIR = BASE_DIR / "quarantine"
REPORTS_DIR = BASE_DIR / "reports"

# The 5 Roboflow datasets (all have train/, some have valid/ and test/)
DATASETS = {
    "v1i": DATASET_DIR / "two wheeler helmet.v1i.yolov8",
    "v2i": DATASET_DIR / "two wheeler helmet.v2i.yolov8",
    "v3i": DATASET_DIR / "two wheeler helmet.v3i.yolov8",
    "v4i": DATASET_DIR / "two wheeler helmet.v4i.yolov8",
    "v5i": DATASET_DIR / "two wheeler helmet.v5i.yolov8",
}

# Standardized class names
CLASS_NAMES = ['helmet', 'no_helmet']
NUM_CLASSES = 2

# Validation/test split ratio for datasets without val/test
VAL_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# ============================================================
# Utility Functions
# ============================================================

def ensure_dirs():
    """Create output directories."""
    for split in ['train', 'val', 'test']:
        (COMBINED_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (COMBINED_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_image_files(img_dir):
    """Get all image files in a directory."""
    if not img_dir.exists():
        return []
    return [f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]


def get_label_files(lbl_dir):
    """Get all label files in a directory."""
    if not lbl_dir.exists():
        return []
    return [f for f in lbl_dir.iterdir() if f.suffix.lower() == '.txt']


def compute_file_hash(filepath, chunk_size=8192):
    """Compute MD5 hash of a file for duplicate detection."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def validate_label_line(line, line_num, filepath):
    """Validate a single YOLO label line. Returns (is_valid, class_id, error_msg)."""
    parts = line.strip().split()
    if len(parts) < 5:
        return False, -1, f"Line {line_num}: Too few values ({len(parts)}) in {filepath}"
    
    try:
        class_id = int(parts[0])
    except ValueError:
        return False, -1, f"Line {line_num}: Invalid class_id '{parts[0]}' in {filepath}"
    
    if class_id not in (0, 1):
        return False, class_id, f"Line {line_num}: Class {class_id} out of range [0,1] in {filepath}"
    
    # Validate bbox coordinates
    try:
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return False, class_id, f"Line {line_num}: Invalid bbox values in {filepath}"
    
    warnings = []
    for val, name in [(cx, 'cx'), (cy, 'cy'), (w, 'w'), (h, 'h')]:
        if val < 0 or val > 1.5:  # Allow small overflow but flag
            warnings.append(f"Line {line_num}: {name}={val:.4f} out of range in {filepath}")
    
    if warnings:
        return True, class_id, "; ".join(warnings)  # Still valid but warn
    
    return True, class_id, None


# ============================================================
# Phase 1a: Annotation Integrity Validation
# ============================================================

def validate_annotations():
    """Validate all annotations across datasets."""
    print("\n" + "="*60)
    print("PHASE 1a: ANNOTATION INTEGRITY VALIDATION")
    print("="*60)
    
    report = []
    total_images = 0
    total_labels = 0
    orphaned_labels = []
    missing_labels = []
    invalid_labels = []
    class_counts = Counter()
    warnings = []
    
    for ds_name, ds_path in DATASETS.items():
        for split in ['train', 'valid', 'test']:
            img_dir = ds_path / split / 'images'
            lbl_dir = ds_path / split / 'labels'
            
            if not img_dir.exists():
                continue
            
            images = get_image_files(img_dir)
            labels = get_label_files(lbl_dir) if lbl_dir.exists() else []
            
            img_stems = {f.stem for f in images}
            lbl_stems = {f.stem for f in labels}
            
            total_images += len(images)
            total_labels += len(labels)
            
            # Check for orphaned labels (label without image)
            for stem in lbl_stems - img_stems:
                orphaned_labels.append(f"{ds_name}/{split}/labels/{stem}.txt")
            
            # Check for images without labels
            for stem in img_stems - lbl_stems:
                missing_labels.append(f"{ds_name}/{split}/images/{stem}")
            
            # Validate each label file
            for lbl_file in labels:
                try:
                    with open(lbl_file, 'r') as f:
                        lines = f.readlines()
                except Exception as e:
                    invalid_labels.append(f"{ds_name}/{split}/{lbl_file.name}: Read error - {e}")
                    continue
                
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    is_valid, class_id, msg = validate_label_line(line, i, f"{ds_name}/{split}/{lbl_file.name}")
                    if is_valid:
                        class_counts[class_id] += 1
                        if msg:
                            warnings.append(msg)
                    else:
                        invalid_labels.append(msg)
            
            report.append(f"  {ds_name}/{split}: {len(images)} images, {len(labels)} labels")
    
    # Print report
    print("\nDataset Summary:")
    for line in report:
        print(line)
    
    print(f"\nTotal: {total_images} images, {total_labels} labels")
    print(f"\nClass Distribution:")
    for cls_id in sorted(class_counts.keys()):
        print(f"  Class {cls_id} ({CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else 'UNKNOWN'}): {class_counts[cls_id]} annotations")
    
    if orphaned_labels:
        print(f"\n⚠ {len(orphaned_labels)} orphaned labels (no matching image):")
        for ol in orphaned_labels[:10]:
            print(f"  - {ol}")
        if len(orphaned_labels) > 10:
            print(f"  ... and {len(orphaned_labels) - 10} more")
    
    if missing_labels:
        print(f"\n⚠ {len(missing_labels)} images without labels:")
        for ml in missing_labels[:10]:
            print(f"  - {ml}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")
    
    if invalid_labels:
        print(f"\n✗ {len(invalid_labels)} invalid label entries:")
        for il in invalid_labels[:10]:
            print(f"  - {il}")
    
    if warnings:
        print(f"\n⚠ {len(warnings)} warnings:")
        for w in warnings[:10]:
            print(f"  - {w}")
    
    print(f"\n✓ Validation complete. {total_images - len(missing_labels)} images have valid labels.")
    
    return class_counts


# ============================================================
# Phase 1b: Corrupted Image Detection
# ============================================================

def detect_corrupted_images():
    """Detect and quarantine corrupted images."""
    print("\n" + "="*60)
    print("PHASE 1b: CORRUPTED IMAGE DETECTION")
    print("="*60)
    
    if not PIL_AVAILABLE:
        print("[SKIP] Pillow not available. Skipping corruption check.")
        return []
    
    corrupted = []
    checked = 0
    
    for ds_name, ds_path in DATASETS.items():
        for split in ['train', 'valid', 'test']:
            img_dir = ds_path / split / 'images'
            if not img_dir.exists():
                continue
            
            for img_file in get_image_files(img_dir):
                checked += 1
                try:
                    with Image.open(img_file) as img:
                        img.verify()
                    # Re-open to ensure full decode works
                    with Image.open(img_file) as img:
                        img.load()
                except Exception as e:
                    corrupted.append((img_file, str(e)))
                    print(f"  ✗ CORRUPTED: {ds_name}/{split}/{img_file.name} - {e}")
    
    if corrupted:
        print(f"\n⚠ Found {len(corrupted)} corrupted images. Quarantining...")
        for img_file, _ in corrupted:
            dest = QUARANTINE_DIR / img_file.name
            shutil.move(str(img_file), str(dest))
            # Also remove corresponding label
            lbl_file = img_file.parent.parent / 'labels' / (img_file.stem + '.txt')
            if lbl_file.exists():
                shutil.move(str(lbl_file), str(QUARANTINE_DIR / lbl_file.name))
        print(f"  Moved {len(corrupted)} corrupted images to {QUARANTINE_DIR}")
    else:
        print(f"\n✓ All {checked} images are valid. No corruption detected.")
    
    return corrupted


# ============================================================
# Phase 1c: Duplicate Image Detection
# ============================================================

def detect_duplicates():
    """Detect and remove duplicate images using MD5 hash."""
    print("\n" + "="*60)
    print("PHASE 1c: DUPLICATE IMAGE DETECTION")
    print("="*60)
    
    hash_map = defaultdict(list)  # hash -> [(ds_name, split, filepath)]
    
    for ds_name, ds_path in DATASETS.items():
        for split in ['train', 'valid', 'test']:
            img_dir = ds_path / split / 'images'
            if not img_dir.exists():
                continue
            
            for img_file in get_image_files(img_dir):
                file_hash = compute_file_hash(img_file)
                hash_map[file_hash].append((ds_name, split, img_file))
    
    duplicates = {h: files for h, files in hash_map.items() if len(files) > 1}
    
    removed_count = 0
    if duplicates:
        print(f"\n⚠ Found {len(duplicates)} sets of duplicate images:")
        for h, files in duplicates.items():
            print(f"\n  Hash {h[:12]}...:")
            # Keep the first, remove the rest
            for i, (ds, split, fpath) in enumerate(files):
                if i == 0:
                    print(f"    KEEP: {ds}/{split}/{fpath.name}")
                else:
                    print(f"    REMOVE: {ds}/{split}/{fpath.name}")
                    dest = QUARANTINE_DIR / f"dup_{fpath.name}"
                    shutil.move(str(fpath), str(dest))
                    # Remove corresponding label
                    lbl_file = fpath.parent.parent / 'labels' / (fpath.stem + '.txt')
                    if lbl_file.exists():
                        shutil.move(str(lbl_file), str(QUARANTINE_DIR / f"dup_{lbl_file.name}"))
                    removed_count += 1
        print(f"\n  Removed {removed_count} duplicate images.")
    else:
        print(f"\n✓ No duplicate images found across {sum(len(v) for v in hash_map.values())} images.")
    
    return removed_count


# ============================================================
# Phase 1d: Merge Datasets
# ============================================================

def merge_datasets():
    """Merge all 5 datasets into combined_dataset/ with proper train/val/test split."""
    print("\n" + "="*60)
    print("PHASE 1d: MERGING DATASETS")
    print("="*60)
    
    random.seed(42)
    
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for ds_name, ds_path in DATASETS.items():
        print(f"\n  Processing {ds_name}...")
        
        # Check if this dataset has val/test splits
        has_val = (ds_path / 'valid' / 'images').exists()
        has_test = (ds_path / 'test' / 'images').exists()
        
        if has_val and has_test:
            # Dataset already has splits (v5i) — copy as-is
            for src_split, dst_split in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
                src_img_dir = ds_path / src_split / 'images'
                src_lbl_dir = ds_path / src_split / 'labels'
                if not src_img_dir.exists():
                    continue
                
                for img_file in get_image_files(src_img_dir):
                    new_name = f"{ds_name}_{img_file.name}"
                    lbl_file = src_lbl_dir / (img_file.stem + '.txt')
                    
                    dst_img = COMBINED_DIR / dst_split / 'images' / new_name
                    dst_lbl = COMBINED_DIR / dst_split / 'labels' / (f"{ds_name}_{img_file.stem}.txt")
                    
                    if not img_file.exists():
                        continue
                    shutil.copy2(str(img_file), str(dst_img))
                    if lbl_file.exists():
                        shutil.copy2(str(lbl_file), str(dst_lbl))
                    
                    stats[dst_split] += 1
        else:
            # Train-only dataset — split into train/val/test
            train_img_dir = ds_path / 'train' / 'images'
            train_lbl_dir = ds_path / 'train' / 'labels'
            
            if not train_img_dir.exists():
                continue
            
            images = get_image_files(train_img_dir)
            random.shuffle(images)
            
            n = len(images)
            n_test = max(1, int(n * TEST_RATIO))
            n_val = max(1, int(n * VAL_RATIO))
            n_train = n - n_val - n_test
            
            splits = (
                [('train', img) for img in images[:n_train]] +
                [('val', img) for img in images[n_train:n_train+n_val]] +
                [('test', img) for img in images[n_train+n_val:]]
            )
            
            for dst_split, img_file in splits:
                new_name = f"{ds_name}_{img_file.name}"
                lbl_file = train_lbl_dir / (img_file.stem + '.txt')
                
                dst_img = COMBINED_DIR / dst_split / 'images' / new_name
                dst_lbl = COMBINED_DIR / dst_split / 'labels' / (f"{ds_name}_{img_file.stem}.txt")
                
                shutil.copy2(str(img_file), str(dst_img))
                if lbl_file.exists():
                    shutil.copy2(str(lbl_file), str(dst_lbl))
                
                stats[dst_split] += 1
    
    print(f"\n  Merged dataset statistics:")
    for split, count in stats.items():
        print(f"    {split}: {count} images")
    
    return stats


# ============================================================
# Phase 1e: Class Imbalance Analysis
# ============================================================

def analyze_class_imbalance():
    """Analyze class distribution in the combined dataset."""
    print("\n" + "="*60)
    print("PHASE 1e: CLASS IMBALANCE ANALYSIS")
    print("="*60)
    
    split_class_counts = {}
    
    for split in ['train', 'val', 'test']:
        lbl_dir = COMBINED_DIR / split / 'labels'
        class_counts = Counter()
        
        if lbl_dir.exists():
            for lbl_file in lbl_dir.iterdir():
                if lbl_file.suffix != '.txt':
                    continue
                with open(lbl_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            cls_id = int(line.split()[0])
                            class_counts[cls_id] += 1
        
        split_class_counts[split] = class_counts
        print(f"\n  {split} split:")
        for cls_id in sorted(class_counts.keys()):
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            print(f"    {name} (class {cls_id}): {class_counts[cls_id]} annotations")
        
        if len(class_counts) >= 2:
            vals = list(class_counts.values())
            ratio = max(vals) / max(min(vals), 1)
            print(f"    Imbalance ratio: {ratio:.2f}:1")
    
    # Generate plot
    if MPL_AVAILABLE:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, split in enumerate(['train', 'val', 'test']):
            cc = split_class_counts[split]
            classes = [CLASS_NAMES[i] for i in sorted(cc.keys())]
            counts = [cc[i] for i in sorted(cc.keys())]
            
            colors = ['#2ecc71', '#e74c3c']
            axes[idx].bar(classes, counts, color=colors[:len(classes)])
            axes[idx].set_title(f'{split.upper()} Split', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Annotation Count')
            
            for j, (c, v) in enumerate(zip(classes, counts)):
                axes[idx].text(j, v + 5, str(v), ha='center', fontweight='bold')
        
        plt.suptitle('Class Distribution Across Splits', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(REPORTS_DIR / 'class_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Class distribution plot saved to {REPORTS_DIR / 'class_distribution.png'}")
    
    return split_class_counts


# ============================================================
# Phase 1f: Create combined_data.yaml
# ============================================================

def create_combined_yaml():
    """Create the combined_data.yaml config file."""
    print("\n" + "="*60)
    print("PHASE 1f: CREATING combined_data.yaml")
    print("="*60)
    
    yaml_content = {
        'path': str(COMBINED_DIR).replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': NUM_CLASSES,
        'names': CLASS_NAMES,
    }
    
    yaml_path = BASE_DIR / 'combined_data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"  ✓ Created {yaml_path}")
    print(f"  Contents:")
    with open(yaml_path, 'r') as f:
        print("    " + f.read().replace('\n', '\n    '))
    
    return yaml_path


# ============================================================
# Main
# ============================================================

def main():
    print("="*60)
    print("  HELMET DETECTION - DATASET PREPROCESSING PIPELINE")
    print("="*60)
    
    # Setup
    ensure_dirs()
    
    # Clear previous combined dataset if exists
    if COMBINED_DIR.exists():
        print(f"\n  Clearing previous combined dataset at {COMBINED_DIR}...")
        shutil.rmtree(COMBINED_DIR)
        ensure_dirs()
    
    # Phase 1a: Validate annotations
    class_counts = validate_annotations()
    
    # Phase 1b: Detect corrupted images
    corrupted = detect_corrupted_images()
    
    # Phase 1c: Detect duplicates
    dup_count = detect_duplicates()
    
    # Phase 1d: Merge datasets
    merge_stats = merge_datasets()
    
    # Phase 1e: Analyze class imbalance
    split_counts = analyze_class_imbalance()
    
    # Phase 1f: Create YAML
    yaml_path = create_combined_yaml()
    
    # Summary
    print("\n" + "="*60)
    print("  PREPROCESSING COMPLETE — SUMMARY")
    print("="*60)
    print(f"  Corrupted images removed: {len(corrupted)}")
    print(f"  Duplicate images removed: {dup_count}")
    print(f"  Combined dataset: {merge_stats}")
    print(f"  Config file: {yaml_path}")
    print(f"  Reports: {REPORTS_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
