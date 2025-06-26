# pest_detection_train.py
import os
import tensorflow as tf
from keras import layers, models
from keras.utils import image_dataset_from_directory
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PestModelTrainer:
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.model = None
        self.class_names = []
        self.history = None

    def load_datasets(self, train_dir: str, test_dir: str, batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load training and testing datasets from directories"""
        logger.info("Loading datasets...")
        
        # Load only 4 samples per class
        train_dataset = image_dataset_from_directory(
            train_dir,
            image_size=self.image_size,
            batch_size=batch_size,
            label_mode='int',
            shuffle=True,
            seed=42,
            validation_split=0.8,  # Take only 20% of data (which will be ~4 samples per class if you have 20+ samples)
            subset='training'
        )
        
        # Load only 4 samples per class for testing
        test_dataset = image_dataset_from_directory(
            test_dir,
            image_size=self.image_size,
            batch_size=batch_size,
            label_mode='int',
            shuffle=True,
            seed=42,
            validation_split=0.8,  # Take only 20% of data
            subset='validation'
        )
        
        self.class_names = train_dataset.class_names
        logger.info(f"Class names: {self.class_names}")
        logger.info(f"Training samples: {len(train_dataset.file_paths)}")
        logger.info(f"Testing samples: {len(test_dataset.file_paths)}")
        
        return train_dataset, test_dataset

    def build_model(self) -> models.Sequential:
        """Build the CNN model architecture"""
        logger.info("Building model...")
        
        model = models.Sequential([
            layers.Input(shape=(*self.image_size, 3)),
            layers.Rescaling(1./255),
            layers.Conv2D(32, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.class_names), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_model(self, train_dataset: tf.data.Dataset, 
                   test_dataset: tf.data.Dataset, 
                   epochs: int = 10,  # Changed to 10 epochs
                   output_dir: str = "models") -> None:
        """Train the pest detection model"""
        logger.info("Training model...")
        
        self.model = self.build_model()
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(output_dir, "pest_detection_model.keras"),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=3,  # Reduced patience
                restore_best_weights=True
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Save class names
        np.save(os.path.join(output_dir, "pest_class_names.npy"), self.class_names)

    def plot_training_history(self) -> None:
        """Plot training and validation metrics"""
        if self.history is None:
            logger.warning("No training history available")
            return
            
        plt.figure(figsize=(12, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

def main():
    # Configuration
    train_dir = "/workspaces/AgriTech-Hackathon/agricultural_ai_system/Model training/datasets/pest/pest/train"
    test_dir = "/workspaces/AgriTech-Hackathon/agricultural_ai_system/Model training/datasets/pest/pest/test"
    output_dir = "models"
    epochs = 10  # Set to 10 epochs
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and train
    trainer = PestModelTrainer()
    train_dataset, test_dataset = trainer.load_datasets(train_dir, test_dir)
    trainer.train_model(train_dataset, test_dataset, epochs, output_dir)
    trainer.plot_training_history()
    
    logger.info("Training complete. Model saved to %s", output_dir)

if __name__ == "__main__":
    main()