# nutrient_analysis.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from typing import Any, Tuple, Dict, List, Optional
import logging
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NutrientAnalyzer:
    _instance = None
    
    def __init__(self, model_path: str = None, class_names_path: str = None):
        self.model = None
        self.class_names = []
        self.image_size = (256, 256)
        
        if model_path and class_names_path:
            self.load_model(model_path, class_names_path)

    @classmethod
    def get_instance(cls, 
                    model_path: str = "/workspaces/AI-Based-Crop-Monitoring/models/nutrient_model.keras",
                    class_names_path: str = "/workspaces/AI-Based-Crop-Monitoring/models/nutrient_class_names.npy"):
        if cls._instance is None:
            cls._instance = cls(model_path=model_path, class_names_path=class_names_path)
            logger.info("Initialized new NutrientAnalyzer instance")
        return cls._instance

    def load_model(self, model_path: str, class_names_path: str) -> None:
        """
        Load a pre-trained nutrient deficiency model and class names.
        
        Args:
            model_path: Path to the model file (.keras)
            class_names_path: Path to class names file (.npy)
            
        Raises:
            ValueError: If model or class names cannot be loaded
        """
        try:
            self.model = load_model(model_path)
            self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
            logger.info(f"Loaded nutrient analysis model from {model_path}")
            logger.info(f"Class names: {self.class_names}")
        except Exception as e:
            logger.error(f"Failed to load nutrient analysis model: {str(e)}")
            raise ValueError(f"Could not load model: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for nutrient analysis.
        
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
                
            # Resize and preprocess for EfficientNet
            image = cv2.resize(image, self.image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image = keras.applications.efficientnet.preprocess_input(image)
            return np.expand_dims(image, axis=0)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def generate_heatmap(self, img_array: np.ndarray, pred_index: int) -> Optional[np.ndarray]:
        """
        Generate Grad-CAM heatmap for visualization.
        
        Args:
            img_array: Preprocessed image array
            pred_index: Predicted class index
            
        Returns:
            Heatmap array or None if failed
        """
        try:
            # Get the last convolutional layer
            last_conv_layer = None
            for layer in self.model.layers[::-1]:
                if isinstance(layer, keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break
            
            if last_conv_layer is None:
                raise ValueError("No convolutional layer found in model")
            
            # Create gradient model
            grad_model = keras.models.Model(
                self.model.inputs,
                [self.model.get_layer(last_conv_layer).output, self.model.output]
            )
            
            # Compute gradient
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, pred_index]
                
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Generate heatmap
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
            
        except Exception as e:
            logger.warning(f"Could not generate heatmap: {str(e)}")
            return None

    def analyze_leaf(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a leaf image for nutrient deficiencies.
        
        Args:
            image_path: Path to the leaf image file
            
        Returns:
            Dictionary containing analysis results and visualizations
            
        Raises:
            ValueError: If image cannot be processed or model not loaded
        """
        if self.model is None:
            raise ValueError("Nutrient analysis model not loaded")
            
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            original_img = np.array(img)
            img_array = self.preprocess_image(original_img)
            
            # Make prediction
            predictions = self.model.predict(img_array)
            scores = tf.nn.softmax(predictions[0]).numpy()
            predicted_idx = np.argmax(scores)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(scores[predicted_idx] * 100)
            
            # Generate visualizations
            fig = self._create_visualization(original_img, scores, predicted_class, confidence, img_array)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(predicted_class, scores)
            
            # Convert visualization to base64
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                "status": "success",
                "primary_deficiency": predicted_class,
                "confidence": confidence,
                "all_deficiencies": {name: float(score*100) for name, score in zip(self.class_names, scores)},
                "recommendations": recommendations,
                "visualization": img_base64
            }
            
        except Exception as e:
            logger.error(f"Error analyzing leaf image: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _create_visualization(self, original_img: np.ndarray, scores: np.ndarray, 
                            predicted_class: str, confidence: float, 
                            img_array: np.ndarray) -> plt.Figure:
        """Create the visualization figure with subplots"""
        fig = plt.figure(figsize=(18, 6))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title(f"Input Leaf\nPredicted: {predicted_class}\nConfidence: {confidence:.1f}%")
        plt.axis('off')
        
        # Confidence scores
        plt.subplot(1, 3, 2)
        df = pd.DataFrame({
            'Deficiency': self.class_names,
            'Confidence (%)': (scores * 100).round(1)
        }).sort_values('Confidence (%)', ascending=False)
        
        sns.barplot(data=df, x='Confidence (%)', y='Deficiency', palette='viridis')
        plt.title('Nutrient Deficiency Probabilities')
        plt.xlim(0, 100)
        
        # Grad-CAM heatmap
        plt.subplot(1, 3, 3)
        try:
            heatmap = self.generate_heatmap(img_array, np.argmax(scores))
            if heatmap is not None:
                heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
                heatmap = heatmap.resize(original_img.shape[:2][::-1], Image.LANCZOS)
                heatmap = np.array(heatmap) / 255
                
                plt.imshow(original_img)
                plt.imshow(heatmap, alpha=0.5, cmap='jet')
                plt.title('Attention Heatmap')
            else:
                plt.imshow(original_img)
                plt.title('Original Image (Heatmap failed)')
        except:
            plt.imshow(original_img)
            plt.title('Original Image (Heatmap failed)')
            
        plt.axis('off')
        plt.tight_layout()
        
        return fig

    def _generate_recommendations(self, predicted_class: str, scores: np.ndarray) -> List[str]:
        """Generate recommendations based on the analysis"""
        recommendations = []
        
        if predicted_class == "Healthy":
            recommendations.append("Leaf appears healthy. No action needed.")
        else:
            recommendations.append(f"Primary Action: Address {predicted_class} deficiency")
            
            # Check for secondary deficiencies (>20% confidence)
            secondary = [(n, s) for n, s in zip(self.class_names, scores) 
                        if s > 0.2 and n != predicted_class and n != "Healthy"]
            if secondary:
                recommendations.append("Secondary Potential Issues:")
                for name, score in secondary:
                    recommendations.append(f"- Possible {name} deficiency ({score*100:.1f}% confidence)")
        
        recommendations.append("Note: For severe cases, consult an agronomist for soil testing")
        return recommendations

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 encoded image"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def analyze_leaf_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze leaf image from bytes (useful for web uploads).
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Convert bytes to numpy array
            image = Image.open(io.BytesIO(image_bytes))
            temp_path = "temp_leaf_image.jpg"
            image.save(temp_path)
            
            # Analyze using the file-based method
            result = self.analyze_leaf(temp_path)
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
    def get_nutrient_analyzer():
        return NutrientAnalyzer.get_instance()
    logger.info("Nutrient analyzer accessor initialized")
except Exception as e:
    logger.error(f"Failed to initialize nutrient analyzer accessor: {str(e)}")
    def get_nutrient_analyzer(): return None