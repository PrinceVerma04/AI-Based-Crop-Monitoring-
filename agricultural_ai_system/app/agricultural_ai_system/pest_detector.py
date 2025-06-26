# pest_detection.py
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from typing import Dict, Any, Optional
import logging
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PestDetector:
    _instance = None
    
    def __init__(self, model_path: str = None, class_names_path: str = None):
        self.model = None
        self.class_names = []
        self.input_size = (224, 224)
        
        if model_path and class_names_path:
            self.load_model(model_path, class_names_path)

    @classmethod
    def get_instance(cls,
                    model_path: str = "/workspaces/AI-Based-Crop-Monitoring/models/pest_detection_model.keras",
                    class_names_path: str = "/workspaces/AI-Based-Crop-Monitoring/models/pest_class_names.npy"):
        if cls._instance is None:
            cls._instance = cls(model_path=model_path, class_names_path=class_names_path)
            logger.info("Initialized new PestDetector instance")
        return cls._instance

    def load_model(self, model_path: str, class_names_path: str) -> None:
        """
        Load a pre-trained pest detection model and class names.
        
        Args:
            model_path: Path to the model file
            class_names_path: Path to class names file
            
        Raises:
            ValueError: If model or class names cannot be loaded
        """
        try:
            self.model = load_model(model_path)
            self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
            logger.info(f"Loaded pest detection model from {model_path}")
            logger.info(f"Class names: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Failed to load pest detection model: {str(e)}")
            raise ValueError(f"Could not load model: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for pest detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image ready for model input
            
        Raises:
            ValueError: If image is invalid
        """
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
                
            # Resize and normalize
            image = cv2.resize(image, self.input_size)
            image = img_to_array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def predict_pest(self, image_path: str) -> Dict[str, Any]:
        """
        Detect pests in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results
            
        Raises:
            ValueError: If image cannot be processed or model not loaded
        """
        if self.model is None:
            raise ValueError("Pest detection model not loaded")
            
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from path: {image_path}")
                
            # Preprocess
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            pest_name = self.class_names[predicted_class]
            
            return {
                "pest_detected": pest_name != "healthy",
                "pest_type": pest_name,
                "confidence": confidence,
                "all_predictions": {name: float(score) for name, score in zip(self.class_names, predictions[0])},
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error during pest detection: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def predict_pest_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Detect pests from image bytes (useful for web uploads).
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Convert bytes to numpy array
            image = Image.open(io.BytesIO(image_bytes))
            image = np.array(image)
            
            # Convert to BGR if needed (OpenCV uses BGR)
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            # Save to temp file and use predict_pest
            temp_path = "temp_pest_image.jpg"
            cv2.imwrite(temp_path, image)
            result = self.predict_pest(temp_path)
            os.remove(temp_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image bytes: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

# Update global instance to use lazy loading
try:
    def get_pest_detector():
        return PestDetector.get_instance()
    logger.info("Pest detector accessor initialized")
except Exception as e:
    logger.error(f"Failed to initialize pest detector accessor: {str(e)}")
    def get_pest_detector(): return None