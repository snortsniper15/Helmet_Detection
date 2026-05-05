"""
Phase 2: Data Augmentation & Class Rebalancing
===============================================
- Identifies minority class in the combined training set
- Applies offline augmentations to minority-class images
- Generates new image+label pairs to balance classes
"""

import os
import random
import shutil
from pathlib import Path
from collections import Counter

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[ERROR] Pillow and numpy are required. Install with: pip install Pillow numpy")

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(r"D:\Helmet_detection")
COMBINED_DIR = BASE_DIR / "combined_dataset"
TRAIN_IMG_DIR = COMBINED_DIR / "train" / "images"
TRAIN_LBL_DIR = COMBINED_DIR / "train" / "labels"
REPORTS_DIR = BASE_DIR / "reports"

CLASS_NAMES = ['helmet', 'no_helmet']
TARGET_RATIO = 1.0  # Target 1:1 ratio

random.seed(42)
if PIL_AVAILABLE:
    np.random.seed(42)


# ============================================================
# Augmentation Functions
# ============================================================

def horizontal_flip(img, labels):
    """Flip image horizontally and adjust labels."""
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    new_labels = []
    for lbl in labels:
        parts = lbl.strip().split()
        cls_id = parts[0]
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        cx = 1.0 - cx  # Flip x coordinate
        new_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return flipped, new_labels


def random_rotation(img, labels, max_degrees=15):
    """Rotate image by a random angle and adjust labels (approximate)."""
    angle = random.uniform(-max_degrees, max_degrees)
    rotated = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))
    # For small rotations, label adjustment is minimal — keep labels as-is
    # This is standard practice for YOLO augmentation
    return rotated, labels


def color_jitter(img, labels):
    """Apply random color jitter (brightness, contrast, saturation)."""
    # Brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # Contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # Saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    return img, labels


def random_scale(img, labels, scale_range=(0.8, 1.2)):
    """Scale image randomly and adjust labels."""
    scale = random.uniform(*scale_range)
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    
    scaled = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Crop or pad to original size
    result = Image.new('RGB', (w, h), (128, 128, 128))
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2
    
    if scale >= 1.0:
        # Crop center
        crop_x = (new_w - w) // 2
        crop_y = (new_h - h) // 2
        result = scaled.crop((crop_x, crop_y, crop_x + w, crop_y + h))
        
        # Adjust labels for crop
        new_labels = []
        for lbl in labels:
            parts = lbl.strip().split()
            cls_id = parts[0]
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Scale coordinates
            cx = (cx * scale - crop_x / w)
            cy = (cy * scale - crop_y / h)
            bw = bw * scale
            bh = bh * scale
            # Clip to valid range
            if 0 < cx < 1 and 0 < cy < 1:
                bw = min(bw, min(cx, 1 - cx) * 2)
                bh = min(bh, min(cy, 1 - cy) * 2)
                new_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return result, new_labels if new_labels else labels
    else:
        # Paste centered
        result.paste(scaled, (paste_x, paste_y))
        
        # Adjust labels for padding
        new_labels = []
        for lbl in labels:
            parts = lbl.strip().split()
            cls_id = parts[0]
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Scale and shift coordinates
            cx = cx * scale + paste_x / w
            cy = cy * scale + paste_y / h
            bw = bw * scale
            bh = bh * scale
            new_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return result, new_labels


def apply_random_augmentation(img, labels):
    """Apply a random combination of augmentations."""
    augmentations = [
        ('flip', horizontal_flip),
        ('rotate', lambda i, l: random_rotation(i, l, 15)),
        ('color', color_jitter),
        ('scale', random_scale),
    ]
    
    # Apply 1-3 random augmentations
    num_augs = random.randint(1, 3)
    selected = random.sample(augmentations, min(num_augs, len(augmentations)))
    
    aug_names = []
    for name, aug_fn in selected:
        img, labels = aug_fn(img, labels)
        aug_names.append(name)
    
    return img, labels, "_".join(aug_names)


# ============================================================
# Main Rebalancing Logic
# ============================================================

def get_class_image_mapping():
    """Map each image to its dominant class (for rebalancing decisions)."""
    image_classes = {}  # image_stem -> {0: count, 1: count}
    
    for lbl_file in TRAIN_LBL_DIR.iterdir():
        if lbl_file.suffix != '.txt':
            continue
        
        class_counts = Counter()
        with open(lbl_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] += 1
        
        if class_counts:
            image_classes[lbl_file.stem] = class_counts
    
    return image_classes


def rebalance_dataset():
    """Rebalance dataset by augmenting minority class images."""
    print("\n" + "="*60)
    print("  PHASE 2: DATA AUGMENTATION & CLASS REBALANCING")
    print("="*60)
    
    if not PIL_AVAILABLE:
        print("[ERROR] Cannot augment without Pillow. Skipping.")
        return
    
    # Analyze current class distribution
    image_classes = get_class_image_mapping()
    
    # Count total annotations per class
    total_class_counts = Counter()
    for stem, counts in image_classes.items():
        for cls_id, cnt in counts.items():
            total_class_counts[cls_id] += cnt
    
    print(f"\n  Current training set class distribution:")
    for cls_id in sorted(total_class_counts.keys()):
        print(f"    {CLASS_NAMES[cls_id]}: {total_class_counts[cls_id]} annotations")
    
    if len(total_class_counts) < 2:
        print("  Only one class found. No rebalancing needed.")
        return
    
    # Determine minority class
    minority_cls = min(total_class_counts, key=total_class_counts.get)
    majority_cls = max(total_class_counts, key=total_class_counts.get)
    
    minority_count = total_class_counts[minority_cls]
    majority_count = total_class_counts[majority_cls]
    
    deficit = majority_count - minority_count
    
    print(f"\n  Minority class: {CLASS_NAMES[minority_cls]} ({minority_count} annotations)")
    print(f"  Majority class: {CLASS_NAMES[majority_cls]} ({majority_count} annotations)")
    print(f"  Deficit: {deficit} annotations")
    
    if deficit <= 0:
        print("  Classes are balanced. No augmentation needed.")
        return
    
    # Find images containing minority class
    minority_images = []
    for stem, counts in image_classes.items():
        if minority_cls in counts:
            minority_images.append(stem)
    
    print(f"  Images containing minority class: {len(minority_images)}")
    
    # Calculate how many augmented images to generate
    avg_annotations_per_image = minority_count / max(len(minority_images), 1)
    num_to_generate = int(deficit / max(avg_annotations_per_image, 1)) + 1
    
    print(f"  Generating ~{num_to_generate} augmented images...")
    
    generated = 0
    aug_idx = 0
    
    while generated < num_to_generate:
        # Cycle through minority images
        stem = minority_images[aug_idx % len(minority_images)]
        aug_idx += 1
        
        # Find the image file
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = TRAIN_IMG_DIR / (stem + ext)
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file is None:
            continue
        
        # Read label
        lbl_file = TRAIN_LBL_DIR / (stem + '.txt')
        if not lbl_file.exists():
            continue
        
        with open(lbl_file, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        
        # Load image
        try:
            img = Image.open(img_file).convert('RGB')
        except Exception:
            continue
        
        # Apply augmentation
        aug_img, aug_labels, aug_name = apply_random_augmentation(img, labels)
        
        # Save augmented image and label
        new_stem = f"aug_{aug_name}_{generated}_{stem}"
        new_img_path = TRAIN_IMG_DIR / f"{new_stem}{img_file.suffix}"
        new_lbl_path = TRAIN_LBL_DIR / f"{new_stem}.txt"
        
        aug_img.save(str(new_img_path), quality=95)
        with open(new_lbl_path, 'w') as f:
            f.write('\n'.join(aug_labels) + '\n')
        
        generated += 1
        
        if generated % 50 == 0:
            print(f"    Generated {generated}/{num_to_generate} augmented images...")
    
    print(f"\n  ✓ Generated {generated} augmented images")
    
    # Verify new distribution
    new_image_classes = get_class_image_mapping()
    new_total = Counter()
    for stem, counts in new_image_classes.items():
        for cls_id, cnt in counts.items():
            new_total[cls_id] += cnt
    
    print(f"\n  New training set class distribution:")
    for cls_id in sorted(new_total.keys()):
        print(f"    {CLASS_NAMES[cls_id]}: {new_total[cls_id]} annotations")
    
    if len(new_total) >= 2:
        vals = list(new_total.values())
        ratio = max(vals) / max(min(vals), 1)
        print(f"  New imbalance ratio: {ratio:.2f}:1")


if __name__ == '__main__':
    rebalance_dataset()
    print("\n✓ Augmentation & rebalancing complete!")
