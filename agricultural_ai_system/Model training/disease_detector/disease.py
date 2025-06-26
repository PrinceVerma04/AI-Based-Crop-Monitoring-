import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from keras import layers, models
from keras.preprocessing import image_dataset_from_directory
from keras.applications import ResNet50V2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

class PlantDiseaseModelTrainer:
    def __init__(self, train_path, test_path, img_size=(224, 224), batch_size=32):
        self.train_path = train_path
        self.test_path = test_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None

    def load_and_preprocess_data(self):
        """Load and preprocess data using tf.data API"""
        print("Loading and preprocessing data...")
        
        # Load datasets
        self.train_dataset = image_dataset_from_directory(
            self.train_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size
        )
        
        self.validation_dataset = image_dataset_from_directory(
            self.train_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size
        )
        
        self.test_dataset = image_dataset_from_directory(
            self.test_path,
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Get class names
        self.class_names = self.train_dataset.class_names
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Configure datasets for performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)
        self.validation_dataset = self.validation_dataset.prefetch(buffer_size=AUTOTUNE)
        self.test_dataset = self.test_dataset.prefetch(buffer_size=AUTOTUNE)

    def create_model(self):
        """Create transfer learning model using ResNet50V2"""
        print("Creating ResNet50V2 model...")
        
        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ])
        
        # Base model
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        base_model.trainable = False
        
        # Model architecture
        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        x = data_augmentation(inputs)
        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(len(self.class_names), activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model created successfully!")
        self.model.summary()

    def train_model(self, epochs=30, fine_tune_epochs=15):
        """Train the model with transfer learning approach"""
        print("Starting model training...")
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
            ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
        ]
        
        # Phase 1: Train with frozen base model
        print("Phase 1: Training with frozen base model...")
        history1 = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.validation_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune with unfrozen layers
        print("Phase 2: Fine-tuning with unfrozen layers...")
        self.model.layers[3].trainable = True  # Unfreeze base model
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = self.model.fit(
            self.train_dataset,
            epochs=fine_tune_epochs,
            validation_data=self.validation_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        print("Training completed!")

    def evaluate_model(self):
        """Evaluate model performance on test data"""
        print("Evaluating model...")
        
        # Get true labels
        y_true = np.concatenate([y for x, y in self.test_dataset], axis=0)
        
        # Get predictions
        y_pred = np.argmax(self.model.predict(self.test_dataset), axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        return accuracy, y_pred, y_true

    def save_model(self, filepath='plant_disease_model.keras'):
        """Save the trained model and class names"""
        self.model.save(filepath)
        
        # Save class names to a text file
        with open('class_names.txt', 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        print(f"Model and class names saved to {filepath}")

def main():
    # Set your data paths here
    TRAIN_PATH = "/workspaces/AgriTech-Hackathon/agricultural_ai_system/Model training/datasets/disease/PlantDoc-Dataset/test"
    TEST_PATH = "/workspaces/AgriTech-Hackathon/agricultural_ai_system/Model training/datasets/disease/PlantDoc-Dataset/test"
    
    # Initialize trainer
    trainer = PlantDiseaseModelTrainer(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        img_size=(224, 224),
        batch_size=32
    )
    
    # Load and preprocess data
    trainer.load_and_preprocess_data()
    
    # Create model
    trainer.create_model()
    
    # Train model
    trainer.train_model(epochs=30, fine_tune_epochs=15)
    
    # Evaluate model
    trainer.evaluate_model()
    
    # Save model
    trainer.save_model('plant_disease_model.keras')

if __name__ == "__main__":
    main()