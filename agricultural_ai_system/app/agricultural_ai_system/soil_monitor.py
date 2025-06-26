# soil_classification.py
import os
import cv2
import joblib
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from keras.models import load_model
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SoilImageAnalyzer:
    _instance = None
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224), 
                 model_dir: str = "/workspaces/AI-Based-Crop-Monitoring/models"):
        self.image_size = image_size
        self.model_dir = model_dir
        self.scaler = None
        self.label_encoder = None
        self.classification_model = None
        self.deep_model = None
        
        self.load_models()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            logger.info("Initialized new SoilImageAnalyzer instance")
        return cls._instance


    def load_models(self) -> None:
        """Load pre-trained models and preprocessing objects"""
        try:
            # Load Random Forest model and scaler
            rf_path = os.path.join(self.model_dir, "/workspaces/AI-Based-Crop-Monitoring/models/soil_classifier_rf.pkl")
            scaler_path = os.path.join(self.model_dir, "/workspaces/AI-Based-Crop-Monitoring/models/soil_scaler.pkl")
            encoder_path = os.path.join(self.model_dir, "/workspaces/AI-Based-Crop-Monitoring/models/soil_label_encoder.pkl")
            cnn_path = os.path.join(self.model_dir, "/workspaces/AI-Based-Crop-Monitoring/models/soil_classifier_cnn.h5")
            
            if os.path.exists(rf_path):
                self.classification_model = joblib.load(rf_path)
                logger.info("Loaded Random Forest model")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
                
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
                logger.info("Loaded label encoder")
                
            if os.path.exists(cnn_path):
                self.deep_model = load_model(cnn_path)
                logger.info("Loaded CNN model")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for analysis.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
            
        Raises:
            ValueError: If image doesn't have 3 channels
        """
        if len(image.shape) != 3:
            raise ValueError("Image should have 3 channels (RGB)")
            
        img = cv2.resize(image, self.image_size)
        return img.astype(np.float32) / 255.0

    def predict_soil_type(self, image: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        """
        Predict soil type using both Random Forest and CNN models.
        
        Args:
            image: Input image as numpy array (RGB format)
            
        Returns:
            Tuple of (RF prediction, CNN prediction) or (None, None) if models not loaded
            
        Raises:
            ValueError: If input image is invalid or models not loaded
        """
        if (self.classification_model is None or 
            self.deep_model is None or 
            self.scaler is None or 
            self.label_encoder is None):
            logger.error("Required models or preprocessing objects not loaded")
            return None, None

        try:
            processed_img = self.preprocess_image(image)
            
            # Random Forest prediction
            features = self._extract_features(processed_img)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            clf_pred = self.classification_model.predict(features_scaled)
            clf_label = self.label_encoder.inverse_transform(clf_pred)[0]
            
            # CNN prediction
            dl_pred = self.deep_model.predict(processed_img.reshape(1, *self.image_size, 3))
            dl_label = self.label_encoder.inverse_transform(np.argmax(dl_pred, axis=1))[0]
            
            return clf_label, dl_label
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ValueError(f"Failed to predict soil type: {str(e)}")

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from image (same as training)"""
        # Color features
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        color_features = []
        for i in range(3):
            color_features.extend([
                np.mean(image[:, :, i]), 
                np.std(image[:, :, i]), 
                np.median(image[:, :, i])
            ])
            color_features.extend([
                np.mean(hsv[:, :, i]), 
                np.std(hsv[:, :, i])
            ])
            color_features.extend([
                np.mean(lab[:, :, i]), 
                np.std(lab[:, :, i])
            ])
        
        # Texture features
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        texture_features = [
            np.mean(grad_x), np.std(grad_x),
            np.mean(grad_y), np.std(grad_y),
            np.var(gray), np.mean(gray), np.std(gray)
        ]
        
        return np.array(color_features + texture_features)

    def analyze_soil_image(self, image_path: str) -> Dict[str, Any]:
        """
        Complete soil analysis pipeline for a single image.
        
        Args:
            image_path: Path to the input image file
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        try:
            # Load and validate image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from path: {image_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get predictions
            rf_pred, cnn_pred = self.predict_soil_type(img)
            
            return {
                "classification_prediction": rf_pred,
                "deep_learning_prediction": cnn_pred,
                "image_size": img.shape,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing soil image: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

# Update global instance to use lazy loading
try:
    def get_soil_analyzer():
        return SoilImageAnalyzer.get_instance()
    logger.info("Soil analyzer accessor initialized")
except Exception as e:
    logger.error(f"Failed to initialize soil analyzer accessor: {str(e)}")
    def get_soil_analyzer(): return None