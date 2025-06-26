# nutrient_train.py
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import seaborn as sns
from typing import Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NutrientModelTrainer:
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.model = None
        self.class_names = []
        self.history = None
        self.history_fine = None

    def create_few_shot_dataset(self, dataset_path: str, samples_per_class: int = 4) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
        """Create dataset with limited samples per class"""
        class_names = sorted(os.listdir(dataset_path))
        file_paths = []
        labels = []
        
        logger.info("\nCreating few-shot dataset with configuration:")
        logger.info(f"- Samples per class: {samples_per_class}")
        logger.info(f"- Classes: {class_names}\n")
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(dataset_path, class_name)
            all_files = os.listdir(class_dir)
            selected_files = random.sample(all_files, min(samples_per_class, len(all_files)))
            
            logger.info(f"Class '{class_name}': Using {len(selected_files)} samples")
            
            for fname in selected_files:
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)
        
        def load_and_preprocess_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.image_size)
            image = keras.applications.efficientnet.preprocess_input(image)
            return image
        
        # Create TensorFlow Dataset
        path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
        image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((image_ds, label_ds))
        
        # Split into train and validation (80/20)
        ds_size = len(file_paths)
        train_size = int(0.8 * ds_size)
        
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size)
        
        # Apply augmentation only to training set
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2)
        ])
        
        def augment(image, label):
            image = augmentation(image)
            return image, label
        
        train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch both datasets
        train_ds = train_ds.batch(16).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(16).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, class_names

    def build_model(self, num_classes: int) -> keras.Model:
        """Build EfficientNetV2 model with transfer learning"""
        base_model = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=(*self.image_size, 3),
            pooling='avg'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        inputs = keras.Input(shape=(*self.image_size, 3))
        x = base_model(inputs, training=False)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.0005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model

    def train_model(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, 
                   class_names: List[str], epochs: int = 10, 
                   output_dir: str = "models") -> None:
        """Train and fine-tune the model"""
        num_classes = len(class_names)
        self.class_names = class_names
        self.model = self.build_model(num_classes)
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(output_dir, "nutrient_model.keras"),
                save_best_only=True,
                monitor="val_accuracy",
                mode="max"
            ),
            keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor="val_accuracy"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        logger.info("\nStarting initial training...")
        self.history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )
        
        # Fine-tuning (unfreeze top layers)
        logger.info("\nStarting fine-tuning...")
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze bottom layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.0005/10),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        self.history_fine = self.model.fit(
            train_ds,
            epochs=int(epochs/2),
            validation_data=val_ds,
            callbacks=callbacks
        )
        
        # Save class names
        np.save(os.path.join(output_dir, "nutrient_class_names.npy"), class_names)

    def plot_training_history(self) -> None:
        """Plot training and validation metrics"""
        if self.history is None or self.history_fine is None:
            logger.warning("No training history available")
            return
            
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        fine_acc = self.history_fine.history['accuracy']
        fine_val_acc = self.history_fine.history['val_accuracy']
        fine_loss = self.history_fine.history['loss']
        fine_val_loss = self.history_fine.history['val_loss']
        
        initial_epochs = len(acc)
        total_epochs = initial_epochs + len(fine_acc)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.plot(np.arange(initial_epochs, total_epochs),
                 fine_acc, label='Fine-tuning Training Accuracy')
        plt.plot(np.arange(initial_epochs, total_epochs),
                 fine_val_acc, label='Fine-tuning Validation Accuracy')
        plt.axvline(initial_epochs-1, color='red', linestyle='--')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        
        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.plot(np.arange(initial_epochs, total_epochs),
                 fine_loss, label='Fine-tuning Training Loss')
        plt.plot(np.arange(initial_epochs, total_epochs),
                 fine_val_loss, label='Fine-tuning Validation Loss')
        plt.axvline(initial_epochs-1, color='red', linestyle='--')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.show()

def main():
    # Configuration
    dataset_path = "/workspaces/AgriTech-Hackathon/agricultural_ai_system/Model training/datasets/nutrient/train"
    output_dir = "models"
    samples_per_class = 4
    epochs = 10
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and train
    trainer = NutrientModelTrainer()
    train_ds, val_ds, class_names = trainer.create_few_shot_dataset(dataset_path, samples_per_class)
    trainer.train_model(train_ds, val_ds, class_names, epochs, output_dir)
    trainer.plot_training_history()
    
    logger.info(f"\nTraining complete. Model saved to {output_dir} directory")

if __name__ == "__main__":
    main()