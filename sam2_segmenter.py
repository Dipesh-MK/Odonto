"""
sam2_segmenter.py

Single-anomaly segmentation module using SAM2 (Segment Anything Model 2).
Provides zero-shot interactive prompt segmentation (points or bounding boxes).
Strictly outputs a single binary mask for the user-selected ANOMALY.
No multi-class baggage.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "sam2_checkpoints" / "sam2_hiera_tiny.pt"
MODEL_CONFIG = "sam2_hiera_t.yaml"

class SAM2AnomalySegmenter:
    _instance = None

    def __init__(self, checkpoint=CHECKPOINT_PATH, config=MODEL_CONFIG, device="cpu"):
        assert os.path.exists(checkpoint), f"SAM2 checkpoint not found at {checkpoint}"
        print(f"Loading SAM2 model from {checkpoint} onto {device}...")
        self.device = device
        self.model = build_sam2(config, str(checkpoint), device=device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.current_image_bgr = None
        self.current_shape = None
        print("SAM2 Anomaly Segmenter ready.")

    @classmethod
    def get_instance(cls, device="cpu"):
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    def set_image(self, image_bgr):
        """
        Encodes the image features for fast prompt inference.
        """
        self.current_image_bgr = image_bgr
        self.current_shape = image_bgr.shape[:2]
        # SAM2 expects RGB format
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

    def segment_anomaly(self, point_coords=None, point_labels=None, box=None):
        """
        Segments a single anomaly given user prompt(s).
        
        Args:
            point_coords: List/Array of [x, y] coordinates
            point_labels: List/Array of labels (1 for positive/anomaly, 0 for negative/background)
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            binary_mask: uint8 2D array (0 or 255)
            confidence: float score [0.0, 1.0]
        """
        if self.current_image_bgr is None:
            raise ValueError("No image set. Call set_image() first.")

        pts = None
        lbls = None
        bx = None

        if point_coords is not None and len(point_coords) > 0:
            pts = np.array(point_coords, dtype=np.float32)
            if point_labels is None:
                lbls = np.ones(len(pts), dtype=np.int32)
            else:
                lbls = np.array(point_labels, dtype=np.int32)

        if box is not None and len(box) == 4:
            bx = np.array(box, dtype=np.float32)

        if pts is None and bx is None:
            raise ValueError("Must provide at least one point prompt or bounding box.")

        masks, scores, _ = self.predictor.predict(
            point_coords=pts,
            point_labels=lbls,
            box=bx,
            multimask_output=True
        )

        # Select mask with highest confidence score
        best_idx = int(np.argmax(scores))
        best_mask = (masks[best_idx] > 0).astype(np.uint8) * 255
        confidence = float(scores[best_idx])

        return best_mask, confidence
