"""
Indian Sign Language Model Training Script
==========================================
Features:
- CUDA GPU acceleration
- XLA JIT compilation for faster training
- Mixed precision (FP16) for 2x speed boost
- Threading-based data loading (Windows compatible)
- Large batch size optimization
- Data augmentation
- Best model checkpointing
- Early stopping
- Learning rate scheduling

Usage:
    python train_indian_model.py
"""

import os
import sys
import platform
from multiprocessing import freeze_support

# =========================================================
# CONFIGURATION (before imports to avoid multiprocessing issues)
# =========================================================
DATASET_PATH = r"D:\sign-to-text-and-speech\New folder\model\Indian"
MODEL_SAVE_PATH = r"D:\sign-to-text-and-speech\New folder\model\indian_sign_model.h5"
WEIGHTS_SAVE_PATH = r"D:\sign-to-text-and-speech\New folder\model\indian_sign_weights.h5"
CHECKPOINT_DIR = r"D:\sign-to-text-and-speech\New folder\model\checkpoints"

# Image settings
IMG_SIZE = 128  # Balanced between speed and accuracy
CHANNELS = 3    # RGB images

# Training settings - optimized for GPU
BATCH_SIZE = 64          # Increase if you have more VRAM (e.g., 128, 256)
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Windows doesn't support multiprocessing well with Keras
# Use threading instead
NUM_WORKERS = 4
USE_MULTIPROCESSING = False  # MUST be False on Windows


def configure_gpu():
    """Configure GPU, XLA, and mixed precision"""
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    
    print("=" * 60)
    print("CONFIGURING GPU & XLA...")
    print("=" * 60)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[OK] Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"     - {gpu.name}")
        
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        
        # Enable XLA JIT compilation for faster training
        tf.config.optimizer.set_jit(True)
        print("[OK] XLA JIT compilation enabled")
        
        # Enable mixed precision (FP16) for faster GPU training
        try:
            mixed_precision.set_global_policy('mixed_float16')
            print("[OK] Mixed precision (FP16) enabled - 2x speed boost")
        except Exception as e:
            print(f"[WARNING] Could not enable mixed precision: {e}")
    else:
        print("[WARNING] No GPU found. Training will be slow on CPU.")
        print("          Install CUDA and cuDNN for GPU acceleration.")
    
    return gpus


def build_model(input_shape, num_classes):
    """
    Build an efficient CNN model optimized for sign language recognition.
    Uses depthwise separable convolutions for speed.
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Depthwise separable conv blocks (efficient like MobileNet)
    def separable_block(x, filters, strides=1):
        x = layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x
    
    # Block 1
    x = separable_block(x, 64, strides=1)
    x = separable_block(x, 64, strides=2)
    x = layers.Dropout(0.1)(x)
    
    # Block 2
    x = separable_block(x, 128, strides=1)
    x = separable_block(x, 128, strides=2)
    x = layers.Dropout(0.2)(x)
    
    # Block 3
    x = separable_block(x, 256, strides=1)
    x = separable_block(x, 256, strides=2)
    x = layers.Dropout(0.3)(x)
    
    # Block 4
    x = separable_block(x, 512, strides=1)
    x = separable_block(x, 512, strides=2)
    x = layers.Dropout(0.4)(x)
    
    # Global pooling and classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer (float32 for mixed precision compatibility)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs, name='IndianSignLanguageModel')
    return model


def create_data_generators():
    """Create training and validation data generators"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    print("\n" + "=" * 60)
    print("LOADING DATASET...")
    print("=" * 60)
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VALIDATION_SPLIT,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Don't flip - sign language is directional
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VALIDATION_SPLIT
    )
    
    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Validation data generator
    val_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator


def get_callbacks(timestamp):
    """Create training callbacks"""
    from tensorflow.keras import callbacks
    
    callback_list = [
        # Save best model based on validation accuracy (use .h5 for compatibility)
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, f"best_model_{timestamp}.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        
        # Save best weights separately
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, f"best_weights_{timestamp}.weights.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=0
        ),
        
        # Early stopping to prevent overfitting
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=os.path.join(CHECKPOINT_DIR, f"logs_{timestamp}"),
            histogram_freq=0,  # Disable histogram for speed
            write_graph=False,
            update_freq='epoch'
        ),
    ]
    
    return callback_list


def main():
    """Main training function"""
    import json
    from datetime import datetime
    import tensorflow as tf
    from tensorflow import keras
    
    # Configure GPU
    gpus = configure_gpu()
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"\n[CONFIG]")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Multiprocessing: {USE_MULTIPROCESSING}")
    print(f"  Platform: {platform.system()}")
    
    # Create data generators
    train_generator, val_generator = create_data_generators()
    
    NUM_CLASSES = len(train_generator.class_indices)
    CLASS_NAMES = list(train_generator.class_indices.keys())
    
    print(f"\n[DATASET INFO]")
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Classes: {CLASS_NAMES}")
    
    # Save class names for inference
    class_mapping = {v: k for k, v in train_generator.class_indices.items()}
    with open(os.path.join(os.path.dirname(MODEL_SAVE_PATH), "class_names.json"), "w") as f:
        json.dump({"class_names": CLASS_NAMES, "class_mapping": class_mapping}, f, indent=2)
    print(f"[OK] Class names saved to class_names.json")
    
    # Build model
    print("\n" + "=" * 60)
    print("BUILDING MODEL...")
    print("=" * 60)
    
    model = build_model(
        input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
        num_classes=NUM_CLASSES
    )
    
    # Compile with optimizer
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    model.summary()
    print(f"\n[OK] Model built with {model.count_params():,} parameters")
    
    # Setup callbacks
    print("\n" + "=" * 60)
    print("SETTING UP CALLBACKS...")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callback_list = get_callbacks(timestamp)
    
    print("[OK] Callbacks configured:")
    print("     - ModelCheckpoint (best model & weights)")
    print("     - EarlyStopping (patience=10)")
    print("     - ReduceLROnPlateau")
    print("     - TensorBoard logging")
    
    # Training
    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60)
    print(f"Training on {train_generator.samples} samples")
    print(f"Validating on {val_generator.samples} samples")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {train_generator.samples // BATCH_SIZE}")
    print("=" * 60 + "\n")
    
    # Calculate steps
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = val_generator.samples // BATCH_SIZE
    
    # Train the model (threading mode for Windows compatibility)
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callback_list,
        workers=NUM_WORKERS,
        use_multiprocessing=USE_MULTIPROCESSING,  # False on Windows
        verbose=1
    )
    
    # Save final model
    print("\n" + "=" * 60)
    print("SAVING FINAL MODEL...")
    print("=" * 60)
    
    model.save(MODEL_SAVE_PATH)
    print(f"[OK] Full model saved to: {MODEL_SAVE_PATH}")
    
    model.save_weights(WEIGHTS_SAVE_PATH)
    print(f"[OK] Weights saved to: {WEIGHTS_SAVE_PATH}")
    
    # Training summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    best_val_acc = max(history.history['val_accuracy'])
    best_val_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n[RESULTS]")
    print(f"  Best validation accuracy: {best_val_acc:.4f} (epoch {best_val_epoch})")
    print(f"  Final training accuracy:  {final_train_acc:.4f}")
    print(f"  Final validation accuracy: {final_val_acc:.4f}")
    print(f"\n[SAVED FILES]")
    print(f"  Model: {MODEL_SAVE_PATH}")
    print(f"  Weights: {WEIGHTS_SAVE_PATH}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Class names: class_names.json")
    
    # Plot training history (optional)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(CHECKPOINT_DIR, f"training_history_{timestamp}.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\n[OK] Training plot saved to: {plot_path}")
        
    except ImportError:
        print("\n[INFO] matplotlib not installed - skipping training plot")
    
    print("\n" + "=" * 60)
    print("DONE! Your model is ready for inference.")
    print("=" * 60)
    
    return history


# =========================================================
# ENTRY POINT - Required for Windows multiprocessing
# =========================================================
if __name__ == '__main__':
    # Required for Windows
    freeze_support()
    
    # Run training
    main()
