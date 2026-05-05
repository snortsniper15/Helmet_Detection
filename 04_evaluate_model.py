"""
Phase 4: Model Evaluation & Error Analysis
============================================
- Evaluates best model on validation and test sets
- Generates mAP@0.5, mAP@0.5:0.95, precision, recall, F1-score
- Creates confusion matrix
- Performs error analysis for helmet vs no_helmet misclassifications
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(r"D:\Helmet_detection")
DATA_YAML = BASE_DIR / "combined_data.yaml"
RUNS_DIR = BASE_DIR / "runs" / "helmet_detection"
BEST_WEIGHTS = RUNS_DIR / "weights" / "best.pt"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['helmet', 'no_helmet']


def evaluate_on_split(model, split_name='val'):
    """Evaluate model on a specific data split."""
    print(f"\n{'='*60}")
    print(f"  EVALUATING ON {split_name.upper()} SET")
    print(f"{'='*60}")
    
    results = model.val(
        data=str(DATA_YAML),
        split=split_name,
        imgsz=640,
        batch=16,
        plots=True,
        save_json=True,
        verbose=True,
        project=str(BASE_DIR / "evaluation"),
        name=f"{split_name}_eval",
        exist_ok=True,
    )
    
    # Extract metrics
    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
    }
    
    # F1 score
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0.0
    
    # Per-class metrics
    per_class = {}
    if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
        for i, name in enumerate(CLASS_NAMES):
            if i < len(results.box.ap50):
                per_class[name] = {
                    'ap50': float(results.box.ap50[i]),
                    'ap50_95': float(results.box.ap[i]) if hasattr(results.box, 'ap') and i < len(results.box.ap) else 0,
                }
    
    # Print results
    print(f"\n  --- {split_name.upper()} Results ---")
    print(f"  mAP@0.5:      {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.1f}%)")
    print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f} ({metrics['mAP50_95']*100:.1f}%)")
    print(f"  Precision:     {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    print(f"  Recall:        {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
    print(f"  F1-Score:      {metrics['f1']:.4f} ({metrics['f1']*100:.1f}%)")
    
    if per_class:
        print(f"\n  Per-Class AP@0.5:")
        for name, vals in per_class.items():
            print(f"    {name}: {vals['ap50']:.4f} ({vals['ap50']*100:.1f}%)")
    
    metrics['per_class'] = per_class
    
    return metrics


def run_error_analysis(model):
    """Analyze misclassifications between helmet and no_helmet."""
    print(f"\n{'='*60}")
    print(f"  ERROR ANALYSIS")
    print(f"{'='*60}")
    
    # Run predictions on test set
    test_img_dir = BASE_DIR / "combined_dataset" / "test" / "images"
    test_lbl_dir = BASE_DIR / "combined_dataset" / "test" / "labels"
    
    if not test_img_dir.exists():
        print("  ⚠ Test image directory not found. Skipping error analysis.")
        return {}
    
    # Get predictions
    results = model.predict(
        source=str(test_img_dir),
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=True,
        project=str(BASE_DIR / "evaluation"),
        name="test_predictions",
        exist_ok=True,
    )
    
    # Compare predictions with ground truth
    error_stats = {
        'total_images': 0,
        'helmet_as_no_helmet': 0,  # FN for helmet
        'no_helmet_as_helmet': 0,  # FN for no_helmet
        'missed_detections': 0,
        'false_positives': 0,
        'misclassified_images': [],
    }
    
    for result in results:
        img_name = Path(result.path).stem
        error_stats['total_images'] += 1
        
        # Load ground truth
        gt_file = test_lbl_dir / f"{img_name}.txt"
        gt_classes = set()
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_classes.add(int(parts[0]))
        
        # Get prediction classes
        pred_classes = set()
        if result.boxes is not None and len(result.boxes) > 0:
            for cls_id in result.boxes.cls:
                pred_classes.add(int(cls_id))
        
        # Analyze errors
        if 0 in gt_classes and 0 not in pred_classes:
            error_stats['helmet_as_no_helmet'] += 1
            error_stats['misclassified_images'].append(img_name)
        if 1 in gt_classes and 1 not in pred_classes:
            error_stats['no_helmet_as_helmet'] += 1
            error_stats['misclassified_images'].append(img_name)
        
        if gt_classes and not pred_classes:
            error_stats['missed_detections'] += 1
        if pred_classes and not gt_classes:
            error_stats['false_positives'] += 1
    
    print(f"\n  Error Analysis Results:")
    print(f"    Total test images: {error_stats['total_images']}")
    print(f"    Helmet missed (FN): {error_stats['helmet_as_no_helmet']}")
    print(f"    No_helmet missed (FN): {error_stats['no_helmet_as_helmet']}")
    print(f"    Completely missed detections: {error_stats['missed_detections']}")
    print(f"    False positives (no GT): {error_stats['false_positives']}")
    
    if error_stats['misclassified_images']:
        print(f"\n    Misclassified images ({len(error_stats['misclassified_images'])}):")
        for img in error_stats['misclassified_images'][:20]:
            print(f"      - {img}")
    
    return error_stats


def generate_report(val_metrics, test_metrics, error_stats):
    """Generate a comprehensive evaluation report."""
    report_path = REPORTS_DIR / "evaluation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("  HELMET DETECTION MODEL — EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("MODEL: YOLOv8n\n")
        f.write(f"WEIGHTS: {BEST_WEIGHTS}\n\n")
        
        for split_name, metrics in [('VALIDATION', val_metrics), ('TEST', test_metrics)]:
            f.write(f"\n--- {split_name} SET ---\n")
            f.write(f"  mAP@0.5:      {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.1f}%)\n")
            f.write(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f} ({metrics['mAP50_95']*100:.1f}%)\n")
            f.write(f"  Precision:     {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)\n")
            f.write(f"  Recall:        {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)\n")
            f.write(f"  F1-Score:      {metrics['f1']:.4f} ({metrics['f1']*100:.1f}%)\n")
            
            if metrics.get('per_class'):
                f.write(f"\n  Per-Class AP@0.5:\n")
                for name, vals in metrics['per_class'].items():
                    f.write(f"    {name}: {vals['ap50']:.4f} ({vals['ap50']*100:.1f}%)\n")
        
        if error_stats:
            f.write(f"\n\n--- ERROR ANALYSIS ---\n")
            f.write(f"  Total test images: {error_stats['total_images']}\n")
            f.write(f"  Helmet missed (FN): {error_stats['helmet_as_no_helmet']}\n")
            f.write(f"  No_helmet missed (FN): {error_stats['no_helmet_as_helmet']}\n")
            f.write(f"  Completely missed: {error_stats['missed_detections']}\n")
            f.write(f"  False positives: {error_stats['false_positives']}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"\n  ✓ Evaluation report saved to: {report_path}")
    
    # Also save as JSON
    json_path = REPORTS_DIR / "evaluation_metrics.json"
    json_data = {
        'validation': val_metrics,
        'test': test_metrics,
        'error_analysis': {k: v for k, v in error_stats.items() if k != 'misclassified_images'},
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  ✓ Metrics JSON saved to: {json_path}")


def main():
    print("="*60)
    print("  HELMET DETECTION — MODEL EVALUATION")
    print("="*60)
    
    # Check weights exist
    if not BEST_WEIGHTS.exists():
        print(f"✗ Best weights not found: {BEST_WEIGHTS}")
        print("  Run 03_train_yolov8.py first!")
        sys.exit(1)
    
    # Load best model
    print(f"\n  Loading model: {BEST_WEIGHTS}")
    model = YOLO(str(BEST_WEIGHTS))
    
    # Evaluate on validation set
    val_metrics = evaluate_on_split(model, 'val')
    
    # Evaluate on test set
    test_metrics = evaluate_on_split(model, 'test')
    
    # Error analysis
    error_stats = run_error_analysis(model)
    
    # Generate report
    generate_report(val_metrics, test_metrics, error_stats)
    
    # Check if target mAP > 90% is met
    print(f"\n{'='*60}")
    print(f"  TARGET CHECK")
    print(f"{'='*60}")
    target_map = 0.90
    val_map = val_metrics['mAP50']
    if val_map >= target_map:
        print(f"  ✓ TARGET MET! Val mAP@0.5 = {val_map*100:.1f}% >= {target_map*100:.0f}%")
    else:
        print(f"  ✗ TARGET NOT MET. Val mAP@0.5 = {val_map*100:.1f}% < {target_map*100:.0f}%")
        print(f"    Consider: more data, longer training, larger model (yolov8s/m)")


if __name__ == '__main__':
    main()
