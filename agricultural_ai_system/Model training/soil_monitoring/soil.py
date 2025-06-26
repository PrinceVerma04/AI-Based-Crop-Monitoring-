# soil_classification_train.py
import os
import cv2
import numpy as np
import joblib
import logging
from typing import Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from collections import Counter
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SoilModelTrainer:
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classification_model = None
        self.deep_model = None

    def load_images_from_directory(self, data_dir: str) -> Tuple[List[np.ndarray], List[str]]:
        """Load images and labels from directory structure"""
        images = []
        labels = []
        
        for soil_type in os.listdir(data_dir):
            soil_path = os.path.join(data_dir, soil_type)
            if os.path.isdir(soil_path):
                for img_file in os.listdir(soil_path):
                    img_path = os.path.join(soil_path, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        labels.append(soil_type)
        
        logger.info(f"Loaded {len(images)} images from {data_dir}")
        logger.info(f"Soil types found: {set(labels)}")
        logger.info(f"Label distribution: {Counter(labels)}")
        return images, labels

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for analysis"""
        if len(image.shape) != 3:
            raise ValueError("Image should have 3 channels (RGB)")
            
        img = cv2.resize(image, self.image_size)
        return img.astype(np.float32) / 255.0

    def extract_features(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from images"""
        features = []
        processed_images = []
        
        for img in images:
            processed_img = self.preprocess_image(img)
            processed_images.append(processed_img)
            
            # Color features
            hsv = cv2.cvtColor((processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor((processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            
            color_features = []
            for i in range(3):
                color_features.extend([
                    np.mean(processed_img[:, :, i]), 
                    np.std(processed_img[:, :, i]), 
                    np.median(processed_img[:, :, i])
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
            gray = cv2.cvtColor((processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            texture_features = [
                np.mean(grad_x), np.std(grad_x),
                np.mean(grad_y), np.std(grad_y),
                np.var(gray), np.mean(gray), np.std(gray)
            ]
            
            features.append(color_features + texture_features)
        
        return np.array(features), np.array(processed_images)

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Train and evaluate Random Forest classifier"""
        logger.info("Training Random Forest model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model
        self.classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classification_model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate
        y_pred = self.classification_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        logger.info(f"Random Forest Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:\n" + classification_report(
            y_test_encoded, y_pred, target_names=self.label_encoder.classes_))
        
        return accuracy

    def build_cnn_model(self, num_classes: int) -> Sequential:
        """Build CNN model architecture"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        return model

    def train_cnn(self, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, 
                 epochs: int = 10) -> float:
        """Train and evaluate CNN model"""
        logger.info("Training CNN model...")
        
        # Encode labels
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Build and train model
        num_classes = len(self.label_encoder.classes_)
        self.deep_model = self.build_cnn_model(num_classes)
        
        history = self.deep_model.fit(
            X_train, y_train_encoded,
            validation_data=(X_test, y_test_encoded),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = self.deep_model.evaluate(X_test, y_test_encoded, verbose=0)
        logger.info(f"CNN Test Accuracy: {test_accuracy:.4f}")
        
        return test_accuracy

    def save_models(self, output_dir: str = "models") -> None:
        """Save trained models and preprocessing objects"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Random Forest model and scaler
        joblib.dump(self.classification_model, os.path.join(output_dir, "soil_classifier_rf.pkl"))
        joblib.dump(self.scaler, os.path.join(output_dir, "soil_scaler.pkl"))
        
        # Save CNN model
        self.deep_model.save(os.path.join(output_dir, "soil_classifier_cnn.h5"))
        
        # Save label encoder
        joblib.dump(self.label_encoder, os.path.join(output_dir, "soil_label_encoder.pkl"))
        
        logger.info(f"Models saved to {output_dir} directory")

def main():
    # Configuration
    train_data_dir = "/workspaces/AgriTech-Hackathon/agricultural_ai_system/Model training/datasets/soil/Soil Train/Soil Train"
    test_data_dir = "/workspaces/AgriTech-Hackathon/agricultural_ai_system/Model training/datasets/soil/Soil Test/Soil Test"
    output_dir = "models"
    
    # Initialize trainer
    trainer = SoilModelTrainer()
    
    # Load data
    train_images, train_labels = trainer.load_images_from_directory(train_data_dir)
    test_images, test_labels = trainer.load_images_from_directory(test_data_dir)
    
    # Extract features
    X_train_features, X_train_images = trainer.extract_features(train_images)
    X_test_features, X_test_images = trainer.extract_features(test_images)
    
    # Train models
    rf_accuracy = trainer.train_random_forest(
        X_train_features, train_labels,
        X_test_features, test_labels
    )
    
    cnn_accuracy = trainer.train_cnn(
        X_train_images, train_labels,
        X_test_images, test_labels,
        epochs=10
    )
    
    # Save models
    trainer.save_models(output_dir)
    
    logger.info("\nTraining complete!")
    logger.info(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    logger.info(f"CNN Accuracy: {cnn_accuracy:.4f}")

if __name__ == "__main__":
    main()