"""
Phase 3: YOLOv8 Training
=========================
- Trains YOLOv8n model with pretrained weights
- Optimized hyperparameters (AdamW, cosine LR, early stopping)
- Advanced augmentations (mosaic, mixup, flipping, rotation, color jitter)
- GPU acceleration if available, CPU fallback
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(r"D:\Helmet_detection")
DATA_YAML = BASE_DIR / "combined_data.yaml"
PROJECT_DIR = BASE_DIR / "runs"
EXPERIMENT_NAME = "helmet_detection"

# Detect device
if torch.cuda.is_available():
    DEVICE = 0  # First GPU
    BATCH_SIZE = -1  # Auto-batch for GPU
    EPOCHS = 100
    WORKERS = 8
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
else:
    DEVICE = 'cpu'
    BATCH_SIZE = 8  # Reduced to avoid OOM
    EPOCHS = 15  # Increased to 15 for better accuracy
    WORKERS = 0  # Set to 0 to avoid memory leaks on Windows
    print("⚠ No GPU detected. Training on CPU (slower).")
    print("  Epochs reduced to 50. Install CUDA PyTorch for GPU support.")


def train():
    """Train YOLOv8 model."""
    print("\n" + "="*60)
    print("  PHASE 3: YOLOv8 TRAINING")
    print("="*60)
    
    # Verify data yaml exists
    if not DATA_YAML.exists():
        print(f"✗ Data config not found: {DATA_YAML}")
        print("  Run 01_preprocess_dataset.py first!")
        sys.exit(1)
    
    print(f"\n  Model: YOLOv8n (pretrained)")
    print(f"  Data: {DATA_YAML}")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: 640")
    print(f"  Optimizer: AdamW")
    print(f"  LR schedule: Cosine annealing")
    print(f"  Early stopping patience: 20")
    
    # Load the previously trained weights instead of starting from scratch
    model = YOLO(r'D:\Helmet_detection\models\best.pt')
    
    # Train with optimized hyperparameters
    results = model.train(
        # === Data ===
        data=str(DATA_YAML),
        imgsz=640,
        
        # === Training Schedule ===
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        patience=20,  # Early stopping
        
        # === Optimizer ===
        optimizer='AdamW',
        lr0=0.001,        # Initial learning rate
        lrf=0.01,         # Final LR = lr0 * lrf (cosine decay target)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,      # Cosine LR scheduling
        
        # === Augmentation ===
        mosaic=1.0,       # Mosaic augmentation probability
        mixup=0.15,       # MixUp augmentation probability
        copy_paste=0.0,
        fliplr=0.5,       # Horizontal flip probability
        flipud=0.0,       # No vertical flip (helmets are orientation-sensitive)
        scale=0.5,        # Scale augmentation ±50%
        degrees=10.0,     # Rotation augmentation ±10°
        translate=0.1,    # Translation augmentation
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015,      # Hue jitter
        hsv_s=0.7,        # Saturation jitter
        hsv_v=0.4,        # Value (brightness) jitter
        erasing=0.0,
        
        # === Hardware ===
        device=DEVICE,
        workers=WORKERS,
        
        # === Logging & Saving ===
        project=str(PROJECT_DIR),
        name=EXPERIMENT_NAME,
        exist_ok=True,
        save=True,
        save_period=10,   # Save checkpoint every 10 epochs
        plots=True,       # Generate training plots
        verbose=True,
        
        # === Other ===
        seed=42,
        deterministic=True,
        single_cls=False,  # Multi-class (helmet vs no_helmet)
        rect=False,
        close_mosaic=10,   # Disable mosaic for last 10 epochs
        amp=True if DEVICE != 'cpu' else False,  # Mixed precision for GPU only
        val=True,          # Validate during training
    )
    
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    
    # Print results summary
    run_dir = PROJECT_DIR / EXPERIMENT_NAME
    print(f"\n  Results saved to: {run_dir}")
    print(f"  Best weights: {run_dir / 'weights' / 'best.pt'}")
    print(f"  Last weights: {run_dir / 'weights' / 'last.pt'}")
    
    # Check for training plots
    for plot_name in ['results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png',
                      'F1_curve.png', 'PR_curve.png', 'P_curve.png', 'R_curve.png']:
        plot_path = run_dir / plot_name
        if plot_path.exists():
            print(f"  Plot: {plot_path}")
    
    return results


if __name__ == '__main__':
    train()
