"""
train_3class_seg_balanced.py

Retrains YOLOv8x-seg on the class-balanced 3-class dental pathology dataset:
- Correction: 8x Crack oversampling (bringing Crack:Abrasion/Caries ratio from ~1:47 to ~1:6).
- Extended training: 35 epochs (extended beyond 25 based on Part 1 finding that training had not plateaued).
- Starting Checkpoint: alphadent_weights/yolov8x_AlphaDent_9_classes_640px.pt
- Hardware: Auto-selects CUDA GPU (NVIDIA RTX 4060).
"""

import os
import sys
import time
import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "alphadent_weights" / "yolov8x_AlphaDent_9_classes_640px.pt"
DATA_YAML = BASE_DIR / "dataset_3class_seg_balanced" / "data.yaml"
RUNS_DIR = BASE_DIR / "runs" / "fine_tune_3class_balanced"

def run_training():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Starting weights not found: {WEIGHTS_PATH}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("STARTING CLASS-BALANCED YOLOV8X-SEG FINE-TUNING (35 EPOCHS, 8X CRACK OVERSAMPLE)")
    print("=" * 70)
    print(f"Base Checkpoint : {WEIGHTS_PATH}")
    print(f"Dataset Config  : {DATA_YAML}")
    print(f"Device          : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    model = YOLO(str(WEIGHTS_PATH))

    start_time = time.time()

    results = model.train(
        data=str(DATA_YAML),
        epochs=35,
        imgsz=640,
        batch=4 if device != "cpu" else 2,
        device=device,
        workers=4,
        project=str(RUNS_DIR),
        name="yolov8x_3class_balanced",
        save=True,
        save_period=5,
        val=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.005,
        lrf=0.01,
        patience=15,
        close_mosaic=5,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"BALANCED FINE-TUNING COMPLETED IN {elapsed / 60:.2f} MINUTES")
    print("=" * 70)
    best_weights = RUNS_DIR / "yolov8x_3class_balanced" / "weights" / "best.pt"
    print(f"Best balanced weights saved to: {best_weights}")
    return best_weights

if __name__ == "__main__":
    run_training()
