import os
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List
import logging
from keras.models import load_model
from keras.applications.resnet_v2 import preprocess_input
import cv2
from PIL import Image
import io
import base64
from io import BytesIO
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseDetector:
    _instance = None
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.class_names = []
        self.img_size = (224, 224)
        
        if model_path:
            self.load_model(model_path)

    @classmethod
    def get_instance(cls, model_path: str = "/workspaces/AI-Based-Crop-Monitoring/plant_disease_model.keras"):
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
            logger.info("Initialized new PlantDiseaseDetector instance")
        return cls._instance

    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained disease detection model and its class names.
        
        Args:
            model_path: Path to the .keras model file
            
        Raises:
            ValueError: If model cannot be loaded
        """
        try:
            # Load model
            self.model = load_model(model_path)
            
            # Load class names from adjacent file
            model_dir = os.path.dirname(model_path)
            class_file = os.path.join(model_dir, '/workspaces/AI-Based-Crop-Monitoring/class_names.txt')
            
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                logger.warning("Class names file not found. Using default empty list.")
                self.class_names = []
            
            logger.info(f"Loaded disease detection model from {model_path}")
            logger.info(f"Loaded {len(self.class_names)} class names")
            
        except Exception as e:
            logger.error(f"Failed to load disease detection model: {str(e)}")
            raise ValueError(f"Could not load model: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for disease detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image ready for model input
            
        Raises:
            ValueError: If image cannot be processed
        """
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
                
            # Resize and preprocess for ResNet
            image = cv2.resize(image, self.img_size)
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = preprocess_input(image)
            return np.expand_dims(image, axis=0)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def predict_disease(self, image_path: str) -> Dict[str, Any]:
        """
        Detect plant diseases in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results
            
        Raises:
            ValueError: If image cannot be processed
        """
        if self.model is None:
            raise ValueError("Disease detection model not loaded")
            
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from path: {image_path}")
                
            # Preprocess
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # Handle case where class names might not be loaded
            if len(self.class_names) > 0:
                disease_name = self.class_names[predicted_class]
            else:
                disease_name = f"class_{predicted_class}"
                logger.warning("Using numerical class labels as class names not loaded")
            
            return {
                "disease_detected": disease_name.lower() != "healthy",
                "disease_type": disease_name,
                "confidence": confidence,
                "all_predictions": {name: float(score) for name, score in zip(self.class_names, predictions[0])} if self.class_names else {},
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error during disease detection: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def predict_disease_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Detect plant diseases from image bytes (useful for web uploads).
        
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
                
            # Save to temp file and use predict_disease
            temp_path = "temp_disease_image.jpg"
            cv2.imwrite(temp_path, image)
            result = self.predict_disease(temp_path)
            os.remove(temp_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image bytes: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def generate_visual_report(self, image_path: str) -> Dict[str, Any]:
        """
        Generate a comprehensive visual report for a plant image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results and visualizations
            
        Raises:
            ValueError: If model not loaded or image processing fails
        """
        if self.model is None:
            raise ValueError("Disease detection model not loaded")
            
        if not self.class_names:
            raise ValueError("Class names not loaded - cannot generate report")
            
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            original_img = np.array(img)
            img_array = self.preprocess_image(original_img)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            scores = tf.nn.softmax(predictions[0]).numpy()
            predicted_idx = np.argmax(scores)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(scores[predicted_idx] * 100)
            
            # Generate visualization
            fig = self._create_visualization(original_img, scores, predicted_class, confidence)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(predicted_class)
            
            # Convert visualization to base64
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                "status": "success",
                "disease_detected": predicted_class.lower() != "healthy",
                "disease_type": predicted_class,
                "confidence": confidence,
                "all_predictions": {name: float(score*100) for name, score in zip(self.class_names, scores)},
                "recommendations": recommendations,
                "visualization": img_base64
            }
            
        except Exception as e:
            logger.error(f"Error generating visual report: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _create_visualization(self, original_img: np.ndarray, scores: np.ndarray, 
                            predicted_class: str, confidence: float) -> plt.Figure:
        """Create the visualization figure"""
        fig = plt.figure(figsize=(15, 5))
        
        # Original image with prediction
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.1f}%")
        plt.axis('off')
        
        # Confidence scores
        plt.subplot(1, 2, 2)
        plt.barh(self.class_names, scores * 100, color='skyblue')
        plt.xlabel('Confidence (%)')
        plt.title('Disease Probabilities')
        plt.xlim(0, 100)
        
        plt.tight_layout()
        return fig

    def _generate_recommendations(self, disease_type: str) -> List[str]:
        """Generate treatment recommendations based on disease type"""
        recommendations = []
        disease_lower = disease_type.lower()
        
        if disease_lower == "healthy":
            recommendations.append("Plant appears healthy. No treatment needed.")
        else:
            recommendations.append(f"Detected: {disease_type.replace('_', ' ').title()}")
            recommendations.append("Recommended Actions:")
            
            if "powdery_mildew" in disease_lower:
                recommendations.append("- Apply sulfur or potassium bicarbonate")
                recommendations.append("- Improve air circulation")
                recommendations.append("- Remove severely infected leaves")
            elif "leaf_spot" in disease_lower:
                recommendations.append("- Apply copper-based fungicides")
                recommendations.append("- Remove and destroy infected leaves")
                recommendations.append("- Avoid overhead watering")
            elif "blight" in disease_lower:
                recommendations.append("- Apply chlorothalonil or mancozeb fungicides")
                recommendations.append("- Remove infected plants immediately")
                recommendations.append("- Avoid working with plants when wet")
            elif "rust" in disease_lower:
                recommendations.append("- Apply fungicides containing myclobutanil")
                recommendations.append("- Remove and destroy infected leaves")
                recommendations.append("- Space plants for better air flow")
            elif "bacterial" in disease_lower:
                recommendations.append("- Apply copper-based bactericides")
                recommendations.append("- Prune infected branches")
                recommendations.append("- Disinfect tools between plants")
            else:
                recommendations.append("- Consult a plant pathologist for specific treatment")
                
            recommendations.append("\nFor severe infections, consult a plant pathologist.")
        
        return recommendations

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 encoded image"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

# Update global instance to use lazy loading
try:
    def get_disease_detector():
        return PlantDiseaseDetector.get_instance()
    logger.info("Disease detector accessor initialized")
except Exception as e:
    logger.error(f"Failed to initialize disease detector accessor: {str(e)}")
    def get_disease_detector(): return None