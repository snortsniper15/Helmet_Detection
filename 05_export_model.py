"""
Phase 5: Model Export
======================
- Saves best model weights to models/ directory
- Exports to ONNX format (deployment-ready)
- Exports to TorchScript format
- Benchmarks inference speed
"""

import os
import sys
import time
import shutil
from pathlib import Path

from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(r"D:\Helmet_detection")
RUNS_DIR = BASE_DIR / "runs" / "helmet_detection"
BEST_WEIGHTS = RUNS_DIR / "weights" / "best.pt"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_DIR = BASE_DIR / "combined_dataset"
TEST_IMG_DIR = COMBINED_DIR / "test" / "images"

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def save_best_weights():
    """Copy best weights to models/ directory."""
    print(f"\n{'='*60}")
    print(f"  SAVING BEST WEIGHTS")
    print(f"{'='*60}")
    
    if not BEST_WEIGHTS.exists():
        print(f"  ✗ Best weights not found: {BEST_WEIGHTS}")
        return None
    
    dest = MODELS_DIR / "best.pt"
    shutil.copy2(str(BEST_WEIGHTS), str(dest))
    
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  ✓ Best weights saved to: {dest} ({size_mb:.1f} MB)")
    
    return dest


def export_onnx(model):
    """Export model to ONNX format."""
    print(f"\n{'='*60}")
    print(f"  EXPORTING TO ONNX")
    print(f"{'='*60}")
    
    try:
        onnx_path = model.export(
            format='onnx',
            imgsz=640,
            simplify=True,
            dynamic=False,
            opset=12,
        )
        
        if onnx_path and Path(onnx_path).exists():
            # Move to models directory
            dest = MODELS_DIR / "best.onnx"
            shutil.copy2(str(onnx_path), str(dest))
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✓ ONNX model saved to: {dest} ({size_mb:.1f} MB)")
            return dest
        else:
            print(f"  ✗ ONNX export failed.")
            return None
    except Exception as e:
        print(f"  ✗ ONNX export error: {e}")
        print(f"    Try: pip install onnx onnxruntime")
        return None


def export_torchscript(model):
    """Export model to TorchScript format."""
    print(f"\n{'='*60}")
    print(f"  EXPORTING TO TORCHSCRIPT")
    print(f"{'='*60}")
    
    try:
        ts_path = model.export(
            format='torchscript',
            imgsz=640,
        )
        
        if ts_path and Path(ts_path).exists():
            dest = MODELS_DIR / "best.torchscript"
            shutil.copy2(str(ts_path), str(dest))
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✓ TorchScript model saved to: {dest} ({size_mb:.1f} MB)")
            return dest
        else:
            print(f"  ✗ TorchScript export failed.")
            return None
    except Exception as e:
        print(f"  ✗ TorchScript export error: {e}")
        return None


def benchmark_inference(model, num_images=20):
    """Benchmark inference speed on sample images."""
    print(f"\n{'='*60}")
    print(f"  INFERENCE SPEED BENCHMARK")
    print(f"{'='*60}")
    
    if not TEST_IMG_DIR.exists():
        print("  ⚠ Test images not found. Skipping benchmark.")
        return
    
    # Get sample images
    test_images = [f for f in TEST_IMG_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    if not test_images:
        print("  ⚠ No test images found. Skipping benchmark.")
        return
    
    sample = test_images[:min(num_images, len(test_images))]
    
    # Warmup
    print(f"  Warming up ({len(sample)} images)...")
    for img in sample[:3]:
        model.predict(str(img), imgsz=640, verbose=False)
    
    # Benchmark
    print(f"  Benchmarking on {len(sample)} images...")
    times = []
    for img in sample:
        start = time.perf_counter()
        model.predict(str(img), imgsz=640, verbose=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n  Results ({len(sample)} images):")
    print(f"    Average inference time: {avg_time*1000:.1f} ms")
    print(f"    FPS: {fps:.1f}")
    print(f"    Min: {min_time*1000:.1f} ms")
    print(f"    Max: {max_time*1000:.1f} ms")
    
    if fps >= 30:
        print(f"    ✓ Real-time capable (>= 30 FPS)")
    elif fps >= 15:
        print(f"    ~ Near real-time (15-30 FPS)")
    else:
        print(f"    ⚠ Below real-time (< 15 FPS) — consider GPU or smaller model")


def main():
    print("="*60)
    print("  HELMET DETECTION — MODEL EXPORT")
    print("="*60)
    
    # Check weights
    if not BEST_WEIGHTS.exists():
        print(f"✗ Best weights not found: {BEST_WEIGHTS}")
        print("  Run 03_train_yolov8.py first!")
        sys.exit(1)
    
    # Save best weights
    save_best_weights()
    
    # Load model
    model = YOLO(str(BEST_WEIGHTS))
    
    # Export ONNX
    export_onnx(model)
    
    # Export TorchScript
    export_torchscript(model)
    
    # Benchmark
    benchmark_inference(model)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Models directory: {MODELS_DIR}")
    if MODELS_DIR.exists():
        for f in sorted(MODELS_DIR.iterdir()):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name}: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
