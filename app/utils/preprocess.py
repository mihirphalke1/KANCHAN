"""
Shared image and audio preprocessing utilities.
"""
import io
import numpy as np
from PIL import Image


def load_image_bytes(raw_bytes: bytes, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load image from bytes → RGB numpy array resized to `size`."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def image_to_hsv(rgb_array: np.ndarray) -> np.ndarray:
    """Convert HxWx3 float RGB array to HxWx3 HSV (OpenCV convention)."""
    import cv2
    uint8 = (rgb_array * 255).astype(np.uint8)
    hsv = cv2.cvtColor(uint8, cv2.COLOR_RGB2HSV)
    return hsv.astype(np.float32)


def extract_hsv_stats(rgb_array: np.ndarray) -> dict:
    """Return mean and std for each HSV channel."""
    hsv = image_to_hsv(rgb_array)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return {
        "hue_mean":   float(h.mean()),
        "hue_std":    float(h.std()),
        "sat_mean":   float(s.mean()),
        "sat_std":    float(s.std()),
        "val_mean":   float(v.mean()),
        "val_std":    float(v.std()),
    }
