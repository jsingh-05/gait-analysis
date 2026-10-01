"""
FILE 3: 3_train_classifier.py
PURPOSE: Train CNN + BiLSTM deep learning model for 12-class activity classification
MODEL: CNN + Bidirectional LSTM (proven ~93% accuracy on HuGaDB)
RUN: python 3_train_classifier.py   (run AFTER 1_preprocess.py)
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# TensorFlow / Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dropout, BatchNormalization,
    LSTM, Bidirectional, Dense, Flatten, Input,
    GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── SETTINGS ──────────────────────────────────────────────────────────────────
INPUT_FILE    = os.path.join("processed", "processed_data.csv")
MODEL_FOLDER  = "models"
RESULT_FOLDER = os.path.join("result", "loss_curves")
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Hyperparameters
WINDOW_SIZE  = 128      # sliding window size (timesteps)
STEP_SIZE    = 64       # step between windows
BATCH_SIZE   = 64
EPOCHS       = 50       # EarlyStopping will stop early if needed
LEARNING_RATE = 0.001
TEST_SPLIT   = 0.2
VAL_SPLIT    = 0.1
RANDOM_SEED  = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ Run 1_preprocess.py first!")

    df = pd.read_csv(INPUT_FILE)

    with open(os.path.join("processed", "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    with open(os.path.join("processed", "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"✅ Data loaded: {df.shape}, Features: {len(feature_cols)}, Classes: {len(le.classes_)}")
    return df, feature_cols, le

# ── SLIDING WINDOW ────────────────────────────────────────────────────────────
def create_windows(df, feature_cols, window_size=WINDOW_SIZE, step=STEP_SIZE):
    """Convert time-series rows into fixed-length overlapping windows."""
    print(f"\n🪟 Creating sliding windows (size={window_size}, step={step})...")

    X, y = [], []
    activities = df["label"].values
    features   = df[feature_cols].values

    for start in range(0, len(features) - window_size, step):
        end    = start + window_size
        window = features[start:end]

        # Label = most common activity in this window
        labels_in_window = activities[start:end]
        label = np.bincount(labels_in_window).argmax()

        X.append(window)
        y.append(label)

    X = np.array(X, dtype=np.float32)   # (N, window_size, n_features)
    y = np.array(y, dtype=np.int32)

    print(f"   Windows created: {X.shape[0]:,}")
    print(f"   Shape: {X.shape}")
    return X, y

# ── MODEL 1: CNN + BiLSTM ─────────────────────────────────────────────────────
def build_cnn_bilstm(input_shape, n_classes):
    """
    CNN + Bidirectional LSTM model.
    CNN extracts local spatial features from sensors.
    BiLSTM captures temporal dependencies in both directions.
    """
    model = Sequential([
        # ── CNN Block 1 ──
        Conv1D(64, kernel_size=3, activation="relu", padding="same",
               input_shape=input_shape),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # ── CNN Block 2 ──
        Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # ── CNN Block 3 ──
        Conv1D(256, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # ── BiLSTM Block ──
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.4),

        # ── Classification Head ──
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])

    return model

# ── MODEL 2: Transformer-based (2024 advanced) ────────────────────────────────
def build_transformer_model(input_shape, n_classes):
    """
    CNN + Transformer Encoder (2024 advanced approach).
    Self-attention captures long-range dependencies in sensor data.
    """
    inputs = Input(shape=input_shape)

    # CNN feature extraction
    x = Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    # Transformer Encoder Block 1
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    attn_out = Dropout(0.2)(attn_out)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)

    ff = Dense(256, activation="relu")(x)
    ff = Dense(128)(ff)
    ff = Dropout(0.2)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)

    # Transformer Encoder Block 2
    attn_out2 = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    attn_out2 = Dropout(0.2)(attn_out2)
    x = Add()([x, attn_out2])
    x = LayerNormalization()(x)

    # Global pooling + classifier
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="CNN_Transformer")
    return model

# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train_model(X, y, n_classes, model_type="cnn_bilstm"):
    print(f"\n🏋️  Training {model_type.upper()} model...")

    # One-hot encode labels
    y_cat = to_categorical(y, num_classes=n_classes)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    print(f"   Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build model
    if model_type == "transformer":
        model = build_transformer_model(input_shape, n_classes)
    else:
        model = build_cnn_bilstm(input_shape, n_classes)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Class weights to handle imbalance
    y_int = np.argmax(y_cat, axis=1)
    cw = compute_class_weight("balanced", classes=np.unique(y_int), y=y_int)
    class_weight = dict(enumerate(cw))

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_FOLDER, f"{model_type}_best.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    return model, history, X_test, y_test

# ── PLOT TRAINING CURVES ──────────────────────────────────────────────────────
def plot_training_curves(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2)
    axes[0].set_title(f"{model_name} — Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    best_val_acc = max(history.history["val_accuracy"])
    axes[0].axhline(y=best_val_acc, color="red", linestyle="--", alpha=0.5,
                    label=f"Best Val: {best_val_acc:.4f}")
    axes[0].legend()

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
    axes[1].set_title(f"{model_name} — Loss", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.suptitle(f"Training Curves: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(RESULT_FOLDER, f"{model_name}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📈 Training curves saved → {path}")

    return best_val_acc

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
def save_results(history, model_name):
    """Save training history to CSV for later analysis."""
    hist_df = pd.DataFrame(history.history)
    path = os.path.join(MODEL_FOLDER, f"{model_name}_history.csv")
    hist_df.to_csv(path, index=False)
    print(f"   💾 History saved → {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GAIT PROJECT — Step 3: Training Deep Learning Models")
    print("=" * 60)
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("=" * 60)

    # Load data
    df, feature_cols, le = load_data()
    n_classes = len(le.classes_)

    # Create sliding windows
    X, y = create_windows(df, feature_cols)

    # ── Train Model 1: CNN + BiLSTM ──
    print("\n" + "─" * 50)
    print("  MODEL 1: CNN + Bidirectional LSTM")
    print("─" * 50)
    model1, history1, X_test, y_test = train_model(X, y, n_classes, "cnn_bilstm")
    best1 = plot_training_curves(history1, "CNN_BiLSTM")
    save_results(history1, "cnn_bilstm")

    # Save test data for evaluate.py
    np.save(os.path.join(MODEL_FOLDER, "X_test.npy"), X_test)
    np.save(os.path.join(MODEL_FOLDER, "y_test.npy"), y_test)
    print(f"   💾 Test data saved for evaluation")

    # ── Train Model 2: CNN + Transformer ──
    print("\n" + "─" * 50)
    print("  MODEL 2: CNN + Transformer (2024 Advanced)")
    print("─" * 50)
    model2, history2, _, _ = train_model(X, y, n_classes, "transformer")
    best2 = plot_training_curves(history2, "CNN_Transformer")
    save_results(history2, "transformer")

    # Compare models
    print("\n" + "=" * 60)
    print("  📊 MODEL COMPARISON")
    print("=" * 60)
    print(f"  CNN + BiLSTM   best val accuracy: {best1:.4f} ({best1*100:.2f}%)")
    print(f"  CNN+Transformer best val accuracy: {best2:.4f} ({best2*100:.2f}%)")
    winner = "CNN_BiLSTM" if best1 >= best2 else "CNN_Transformer"
    print(f"  🏆 Best model: {winner}")
    print("=" * 60)
    print("\n▶ Next step: python 4_train_anomaly.py")

if __name__ == "__main__":
    main()
