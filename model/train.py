"""
Model Training Script
Builds, trains, and saves a TensorFlow/Keras model for sign language recognition.
Includes data collection from webcam and training pipeline.
"""

import os
import sys
import json
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from utils.hand_tracker import HandTracker
from utils.preprocessing import extract_features, FEATURE_DIM
from config import SIGN_CLASSES, MODEL_DIR, DATA_DIR

logger = logging.getLogger(__name__)

# ─── Data Collection ────────────────────────────────────────


def collect_training_data(samples_per_class=100):
    """
    Collect hand landmark data from webcam for each sign class.
    Guides the user through showing each gesture.
    
    Args:
        samples_per_class: Number of samples to collect per gesture
    """
    from utils.helpers import ensure_dirs, setup_logging
    setup_logging()

    dataset_dir = os.path.join(DATA_DIR, "collected")
    ensure_dirs(dataset_dir)

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Cannot open camera for data collection")
        return

    all_features = []
    all_labels = []

    for idx, sign in enumerate(SIGN_CLASSES):
        print(f"\n{'='*50}")
        print(f"  Prepare to show gesture: [{sign}]")
        print(f"  Class {idx + 1}/{len(SIGN_CLASSES)}")
        print(f"  Press 'S' to start collecting, 'Q' to quit")
        print(f"{'='*50}")

        # Wait for user to press 'S'
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Show: {sign} | Press 'S' to start",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                tracker.release()
                return

        # Collect samples
        collected = 0
        while collected < samples_per_class:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame, results = tracker.find_hands(frame)
            landmarks = tracker.extract_landmarks(results)

            if landmarks is not None:
                features = extract_features(landmarks)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(idx)
                    collected += 1

            cv2.putText(frame, f"Collecting [{sign}]: {collected}/{samples_per_class}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()

    # Save collected data
    if all_features:
        np.save(os.path.join(dataset_dir, "features.npy"), np.array(all_features))
        np.save(os.path.join(dataset_dir, "labels.npy"), np.array(all_labels))
        logger.info(f"Saved {len(all_features)} samples to {dataset_dir}")
    else:
        logger.warning("No data collected!")


# ─── Model Architecture ─────────────────────────────────────


def build_model(input_dim=FEATURE_DIM, num_classes=None):
    """
    Build a deep neural network for gesture classification.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    if num_classes is None:
        num_classes = len(SIGN_CLASSES)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()
    return model


# ─── LSTM Model Architecture (Optional) ─────────────────────


def build_lstm_model(sequence_length=30, feature_dim=FEATURE_DIM, num_classes=None):
    """
    Build an LSTM model for sequence-based gesture classification.
    Takes sequences of frames instead of single frames for temporal modeling.

    Architecture:
    - Input: (sequence_length, feature_dim) — e.g., 30 frames × 136 features
    - 2× LSTM layers with dropout
    - Dense output with softmax

    Args:
        sequence_length: Number of frames per sequence
        feature_dim: Features per frame
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    if num_classes is None:
        num_classes = len(SIGN_CLASSES)

    model = keras.Sequential([
        layers.Input(shape=(sequence_length, feature_dim)),

        # LSTM layers with recurrent dropout
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),

        layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),

        # Dense classification head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()
    return model


def collect_sequence_data(sequence_length=30, samples_per_class=50):
    """
    Collect sequence training data — captures N consecutive frames per sample.
    Each sample is a sequence of landmarks suitable for LSTM training.

    Args:
        sequence_length: Frames per sequence
        samples_per_class: Number of sequences per gesture class
    """
    from utils.helpers import ensure_dirs, setup_logging
    setup_logging()

    dataset_dir = os.path.join(DATA_DIR, "sequences")
    ensure_dirs(dataset_dir)

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Cannot open camera for sequence data collection")
        return

    all_sequences = []
    all_labels = []

    for idx, sign in enumerate(SIGN_CLASSES):
        print(f"\n{'='*50}")
        print(f"  Prepare gesture: [{sign}]")
        print(f"  Class {idx + 1}/{len(SIGN_CLASSES)}")
        print(f"  Hold the sign steady for {sequence_length} frames")
        print(f"  Press 'S' to start, 'Q' to quit")
        print(f"{'='*50}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Show: {sign} | Press 'S'",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Sequence Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        collected = 0
        while collected < samples_per_class:
            sequence = []
            for _ in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame, results = tracker.find_hands(frame)
                landmarks = tracker.extract_landmarks(results)
                if landmarks is not None:
                    features = extract_features(landmarks)
                    if features is not None:
                        sequence.append(features)

                cv2.putText(frame, f"[{sign}] Seq {collected+1}/{samples_per_class} Frame {len(sequence)}/{sequence_length}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.imshow("Sequence Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if len(sequence) == sequence_length:
                all_sequences.append(sequence)
                all_labels.append(idx)
                collected += 1

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()

    if all_sequences:
        np.save(os.path.join(dataset_dir, "sequences.npy"), np.array(all_sequences))
        np.save(os.path.join(dataset_dir, "seq_labels.npy"), np.array(all_labels))
        logger.info(f"Saved {len(all_sequences)} sequences to {dataset_dir}")


# ─── Training ────────────────────────────────────────────────


def train_model(epochs=50, batch_size=32):
    """
    Train the model on collected data.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    from utils.helpers import ensure_dirs, setup_logging
    setup_logging()
    ensure_dirs(MODEL_DIR)

    dataset_dir = os.path.join(DATA_DIR, "collected")
    features_path = os.path.join(dataset_dir, "features.npy")
    labels_path = os.path.join(dataset_dir, "labels.npy")

    if not os.path.exists(features_path):
        logger.error(f"No training data found at {features_path}")
        logger.info("Run 'python model/train.py --collect' first to collect data")
        return

    # Load data
    X = np.load(features_path)
    y = np.load(labels_path)
    logger.info(f"Loaded {len(X)} samples, {len(np.unique(y))} classes")

    # Shuffle and split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build and train
    num_classes = len(np.unique(y))
    model = build_model(input_dim=X.shape[1], num_classes=num_classes)

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model and labels
    from config import MODEL_PATH, LABELS_PATH
    model.save(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Save label mapping
    used_classes = [SIGN_CLASSES[i] for i in sorted(np.unique(y))]
    with open(LABELS_PATH, "w") as f:
        json.dump(used_classes, f, indent=2)
    logger.info(f"Labels saved to {LABELS_PATH}")

    # Print results
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Validation accuracy: {val_acc:.4f}")


# ─── Generate Demo Model ────────────────────────────────────


def generate_demo_model():
    """
    Generate a pre-trained demo model with synthetic data
    so the system can run immediately without real training.
    """
    from utils.helpers import ensure_dirs, setup_logging
    setup_logging()
    ensure_dirs(MODEL_DIR)

    logger.info("Generating demo model with synthetic data...")

    num_samples_per_class = 50
    num_classes = len(SIGN_CLASSES)

    # Generate synthetic landmark data
    np.random.seed(42)
    X = []
    y = []

    for class_idx in range(num_classes):
        # Create class-specific patterns so model can learn something
        base_pattern = np.random.randn(FEATURE_DIM) * 0.5
        for _ in range(num_samples_per_class):
            sample = base_pattern + np.random.randn(FEATURE_DIM) * 0.15
            X.append(sample)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Build and train quickly
    model = build_model(input_dim=FEATURE_DIM, num_classes=num_classes)

    model.fit(X, y, epochs=20, batch_size=64, validation_split=0.1, verbose=1)

    # Save
    from config import MODEL_PATH, LABELS_PATH
    model.save(MODEL_PATH)
    logger.info(f"Demo model saved to {MODEL_PATH}")

    with open(LABELS_PATH, "w") as f:
        json.dump(SIGN_CLASSES, f, indent=2)
    logger.info(f"Labels saved to {LABELS_PATH}")


# ─── CLI Entry Point ────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sign Language Model Training")
    parser.add_argument("--collect", action="store_true", help="Collect training data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--demo", action="store_true", help="Generate a demo model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--samples", type=int, default=100, help="Samples per class")

    args = parser.parse_args()

    if args.collect:
        collect_training_data(samples_per_class=args.samples)
    elif args.train:
        train_model(epochs=args.epochs)
    elif args.demo:
        generate_demo_model()
    else:
        print("Usage:")
        print("  python train.py --collect   Collect training data via webcam")
        print("  python train.py --train     Train model on collected data")
        print("  python train.py --demo      Generate a demo model (quick start)")
