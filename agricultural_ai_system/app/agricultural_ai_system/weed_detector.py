import os
from ultralytics import YOLO
import cv2
import numpy as np
import logging
from typing import Dict, Any
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeedDetector:
    _instance = None
    
    def __init__(self, model_path: str = "yolov8x-seg.pt"):
        try:
            self.model = YOLO(model_path)  # Store the model as instance variable
            logger.info(f"Weed detection model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load weed detection model: {str(e)}")
            raise ValueError(f"Could not load model: {str(e)}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            try:
                cls._instance = cls()
                logger.info("Initialized new WeedDetector instance")
            except Exception as e:
                logger.error(f"Weed detector initialization failed: {str(e)}")
                cls._instance = None  # Ensure we don't keep trying
        return cls._instance

    def detect_weed(self, image_path: str) -> Dict[str, Any]:
        try:
            # Input validation
            if not os.path.exists(image_path):
                return self._error_response("Image file not found", code="file_not_found")

            img = cv2.imread(image_path)
            if img is None:
                return self._error_response("Invalid image file", code="invalid_image")

            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output_img = img_rgb.copy()
            
            # Run inference
            results = self.model.predict(img_rgb, verbose=False)  # Disable prediction logging
            
            # Process results
            weed_detected = False
            confidence = 0.0
            mask = None
            
            if results and results[0].masks is not None:
                weed_detected = True
                mask = results[0].masks[0].data[0].cpu().numpy()
                
                if results[0].boxes is not None and len(results[0].boxes.conf) > 0:
                    confidence = float(results[0].boxes.conf[0].cpu().numpy())
                
                # Create visualization
                output_img = self._apply_mask(output_img, mask)

            return {
                "status": "success",
                "weed_present": weed_detected,
                "confidence": round(confidence, 2),
                "visualization": output_img,
                "mask_shape": mask.shape if mask is not None else None
            }

        except Exception as e:
            logger.error(f"Weed detection error: {str(e)}", exc_info=True)
            return self._error_response(f"Detection failed: {str(e)}", code="detection_error")

        finally:
            torch.cuda.empty_cache()

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply semi-transparent red mask to detected weeds"""
        try:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            colored_mask = np.zeros_like(image)
            colored_mask[..., 0] = 255  # Red channel
            return (image * (1 - 0.3) + colored_mask * 0.3).astype(np.uint8)
        except Exception as e:
            logger.warning(f"Mask application failed: {str(e)}")
            return image

    def _error_response(self, message: str, code: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "error": message,
            "code": code,
            "weed_present": False,
            "confidence": 0.0,
            "visualization": None
        }

# Safe accessor function
def get_weed_detector():
    try:
        return WeedDetector.get_instance()
    except Exception as e:
        logger.error(f"Failed to get weed detector: {str(e)}")
        return None