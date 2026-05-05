"""
Filter Dataset for Single Class
===============================
- Iterates through all label files in `combined_dataset/train`, `val`, and `test`
- Removes any annotations for class 1 (no_helmet)
- If a label file becomes empty, it's left empty (representing a background image)
"""

import os
from pathlib import Path

BASE_DIR = Path(r"D:\Helmet_detection")
COMBINED_DIR = BASE_DIR / "combined_dataset"

def filter_labels():
    print("="*60)
    print("FILTERING LABELS FOR SINGLE-CLASS DETECTION (HELMET ONLY)")
    print("="*60)
    
    total_removed = 0
    total_files_modified = 0
    total_background_images = 0
    
    for split in ['train', 'val', 'test']:
        lbl_dir = COMBINED_DIR / split / 'labels'
        if not lbl_dir.exists():
            continue
        
        for lbl_file in lbl_dir.iterdir():
            if lbl_file.suffix != '.txt':
                continue
            
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            removed = 0
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id == 0:
                    new_lines.append(line)
                else:
                    removed += 1
            
            if removed > 0:
                with open(lbl_file, 'w') as f:
                    f.writelines(new_lines)
                total_removed += removed
                total_files_modified += 1
                
                if not new_lines:
                    total_background_images += 1

    print(f"\n✓ Removed {total_removed} 'no_helmet' annotations across {total_files_modified} files.")
    print(f"✓ {total_background_images} images are now pure background images (negative samples).")

if __name__ == '__main__':
    filter_labels()
