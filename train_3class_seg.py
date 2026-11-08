"""
train_3class_seg.py

Fine-tunes YOLOv8x-seg on the unified 3-class dental pathology dataset:
- Target Classes: 0: Abrasion, 1: Caries, 2: Crack
- Starting Checkpoint: alphadent_weights/yolov8x_AlphaDent_9_classes_640px.pt
  (Preserves prior intraoral dental photo representation)
- Hardware: Auto-selects CUDA GPU (NVIDIA RTX 4060) if available, with CPU fallback.

Hyperparameter choices:
- imgsz: 640 (standard for YOLOv8 segmentation on intraoral photos)
- epochs: 25 (adequate for fine-tuning pretrained dental weights without overfitting)
- batch: 4 (calibrated for 8GB VRAM of RTX 4060 with YOLOv8x architecture)
- lr0: 0.005, lrf: 0.01 (fine-tuning learning rate, lower than scratch 0.01)
- patience: 10 (early stopping if validation mAP plateaus)
- close_mosaic: 5 (disables mosaic augmentation during final 5 epochs for clean mask boundaries)
"""

import os
import sys
import time
import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "alphadent_weights" / "yolov8x_AlphaDent_9_classes_640px.pt"
DATA_YAML = BASE_DIR / "dataset_3class_seg" / "data.yaml"
RUNS_DIR = BASE_DIR / "runs" / "fine_tune_3class"


def run_training():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Starting weights not found: {WEIGHTS_PATH}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("STARTING YOLOV8X-SEG 3-CLASS DENTAL PATHOLOGY FINE-TUNING")
    print("=" * 70)
    print(f"Base Checkpoint : {WEIGHTS_PATH}")
    print(f"Dataset Config  : {DATA_YAML}")
    print(f"Device          : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    # Load pretrained model
    model = YOLO(str(WEIGHTS_PATH))

    start_time = time.time()

    # Train with transfer learning
    results = model.train(
        data=str(DATA_YAML),
        epochs=25,
        imgsz=640,
        batch=4 if device != "cpu" else 2,
        device=device,
        workers=4,
        project=str(RUNS_DIR),
        name="yolov8x_3class_model",
        save=True,
        save_period=5,
        val=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.005,
        lrf=0.01,
        patience=10,
        close_mosaic=5,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"FINE-TUNING COMPLETED IN {elapsed / 60:.2f} MINUTES")
    print("=" * 70)
    best_weights = RUNS_DIR / "yolov8x_3class_model" / "weights" / "best.pt"
    print(f"Best fine-tuned weights saved to: {best_weights}")
    return best_weights


if __name__ == "__main__":
    run_training()
