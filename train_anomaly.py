"""
FILE 4: 4_train_anomaly.py
PURPOSE: Train anomaly detection model to flag pathological/abnormal gait
METHOD: Autoencoder-based anomaly detection (unsupervised deep learning)
IDEA: Train on NORMAL walking data → anything that reconstructs badly = ABNORMAL
RUN: python 4_train_anomaly.py  (run AFTER 3_train_classifier.py)
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, UpSampling1D, Dense,
    Flatten, Reshape, BatchNormalization, Dropout,
    LSTM, RepeatVector, TimeDistributed, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from scipy import stats

# ── SETTINGS ──────────────────────────────────────────────────────────────────
INPUT_FILE    = os.path.join("processed", "processed_data.csv")
MODEL_FOLDER  = "models"
RESULT_FOLDER = os.path.join("result", "anomaly")
os.makedirs(RESULT_FOLDER, exist_ok=True)

WINDOW_SIZE   = 128
STEP_SIZE     = 64
EPOCHS        = 40
BATCH_SIZE    = 32
LEARNING_RATE = 0.0005
THRESHOLD_PERCENTILE = 95   # top 5% reconstruction error = anomaly
RANDOM_SEED   = 42

# "Normal" activities = activities that don't involve extreme motion patterns
NORMAL_ACTIVITIES = ["Walking", "Running", "Standing", "Sitting",
                     "Going_Up", "Going_Down"]

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("❌ Run 1_preprocess.py first!")

    df = pd.read_csv(INPUT_FILE)

    with open(os.path.join("processed", "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)

    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"✅ Loaded: {df.shape}, Features: {len(feature_cols)}")
    return df, feature_cols

# ── CREATE WINDOWS ────────────────────────────────────────────────────────────
def create_windows(features, window_size=WINDOW_SIZE, step=STEP_SIZE):
    X = []
    for start in range(0, len(features) - window_size, step):
        X.append(features[start:start + window_size])
    return np.array(X, dtype=np.float32)

# ── LSTM AUTOENCODER ──────────────────────────────────────────────────────────
def build_lstm_autoencoder(timesteps, n_features):
    """
    LSTM Autoencoder for anomaly detection.
    Encoder compresses the signal → Decoder reconstructs it.
    High reconstruction error → Abnormal gait!
    """
    inputs = Input(shape=(timesteps, n_features))

    # Encoder
    encoded = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    encoded = Dropout(0.2)(encoded)
    encoded = Bidirectional(LSTM(32, return_sequences=False))(encoded)
    encoded = Dropout(0.2)(encoded)

    # Bottleneck
    bottleneck = Dense(16, activation="relu")(encoded)

    # Decoder
    decoded = RepeatVector(timesteps)(bottleneck)
    decoded = Bidirectional(LSTM(32, return_sequences=True))(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Bidirectional(LSTM(64, return_sequences=True))(decoded)
    decoded = TimeDistributed(Dense(n_features))(decoded)

    autoencoder = Model(inputs, decoded, name="LSTM_Autoencoder")
    return autoencoder

# ── CNN AUTOENCODER (faster alternative) ──────────────────────────────────────
def build_cnn_autoencoder(timesteps, n_features):
    """
    CNN Autoencoder — faster training, good for local patterns.
    """
    inputs = Input(shape=(timesteps, n_features))

    # Encoder
    x = Conv1D(64, 3, activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(32, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(16, 3, activation="relu", padding="same")(x)
    encoded = MaxPooling1D(2)(x)

    # Decoder
    x = Conv1D(16, 3, activation="relu", padding="same")(encoded)
    x = UpSampling1D(2)(x)

    x = Conv1D(32, 3, activation="relu", padding="same")(x)
    x = UpSampling1D(2)(x)

    x = Conv1D(64, 3, activation="relu", padding="same")(x)
    x = UpSampling1D(2)(x)

    # Match original timestep size
    decoded = Conv1D(n_features, 1, activation="linear", padding="same")(x)

    # Crop or pad to match input timestep size
    decoded = decoded[:, :timesteps, :]

    autoencoder = Model(inputs, decoded, name="CNN_Autoencoder")
    return autoencoder

# ── TRAIN AUTOENCODER ─────────────────────────────────────────────────────────
def train_autoencoder(X_normal):
    print(f"\n🏋️  Training LSTM Autoencoder on {X_normal.shape[0]:,} normal windows...")

    X_train, X_val = train_test_split(X_normal, test_size=0.15, random_state=RANDOM_SEED)

    timesteps  = X_train.shape[1]
    n_features = X_train.shape[2]

    autoencoder = build_lstm_autoencoder(timesteps, n_features)
    autoencoder.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    autoencoder.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(
            os.path.join(MODEL_FOLDER, "autoencoder_best.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        )
    ]

    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    return autoencoder, history

# ── COMPUTE THRESHOLD ─────────────────────────────────────────────────────────
def compute_threshold(autoencoder, X_normal):
    """Compute anomaly threshold from reconstruction errors on normal data."""
    X_reconstructed = autoencoder.predict(X_normal, verbose=0)
    errors = np.mean(np.mean(np.square(X_normal - X_reconstructed), axis=2), axis=1)

    threshold = np.percentile(errors, THRESHOLD_PERCENTILE)
    print(f"\n📏 Anomaly Threshold (p{THRESHOLD_PERCENTILE}): {threshold:.6f}")
    print(f"   Mean error (normal): {np.mean(errors):.6f}")
    print(f"   Max  error (normal): {np.max(errors):.6f}")

    return threshold, errors

# ── PLOT AUTOENCODER RESULTS ──────────────────────────────────────────────────
def plot_autoencoder_results(history, normal_errors, threshold):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
    axes[0].set_title("Autoencoder Training Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()

    # Reconstruction error distribution
    axes[1].hist(normal_errors, bins=60, color="steelblue", alpha=0.8, label="Normal Gait")
    axes[1].axvline(threshold, color="red", linewidth=2, linestyle="--",
                    label=f"Threshold = {threshold:.4f}")
    axes[1].set_title("Reconstruction Error Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Reconstruction Error (MSE)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    # Cumulative distribution
    sorted_errors = np.sort(normal_errors)
    cdf = np.arange(len(sorted_errors)) / len(sorted_errors)
    axes[2].plot(sorted_errors, cdf, linewidth=2, color="green")
    axes[2].axvline(threshold, color="red", linewidth=2, linestyle="--",
                    label=f"Threshold ({THRESHOLD_PERCENTILE}th percentile)")
    axes[2].set_title("CDF of Reconstruction Errors", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Reconstruction Error")
    axes[2].set_ylabel("Cumulative Probability")
    axes[2].legend()

    plt.suptitle("LSTM Autoencoder — Anomaly Detection Results",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULT_FOLDER, "autoencoder_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📈 Autoencoder plots saved → {path}")

# ── SIMULATE PATHOLOGICAL DETECTION ──────────────────────────────────────────
def simulate_pathological_detection(autoencoder, X_normal, threshold):
    """
    Since HuGaDB has no pathological labels, we simulate:
    - Add Gaussian noise to normal walking to simulate pathological gait
    - Check if autoencoder correctly flags it as anomaly
    """
    print("\n🔬 Simulating pathological gait detection...")

    # Sample a subset of normal walking
    walking_idx = np.random.choice(len(X_normal), min(200, len(X_normal)), replace=False)
    X_walk = X_normal[walking_idx]

    # Simulate pathological = normal + asymmetric noise (mimic limp)
    noise_scale = 2.0
    X_path = X_walk.copy()
    # Add asymmetric noise to left-side sensors (simulate left-side weakness)
    n_half = X_path.shape[2] // 2
    X_path[:, :, n_half:] += np.random.normal(0, noise_scale,
                                               X_path[:, :, n_half:].shape)

    # Reconstruction errors
    err_normal = np.mean(np.mean(np.square(
        X_walk - autoencoder.predict(X_walk, verbose=0)), axis=2), axis=1)
    err_patho  = np.mean(np.mean(np.square(
        X_path - autoencoder.predict(X_path, verbose=0)), axis=2), axis=1)

    # Classification
    pred_normal = (err_normal > threshold).astype(int)  # 0=normal, 1=anomaly
    pred_patho  = (err_patho  > threshold).astype(int)

    normal_correct = (pred_normal == 0).mean()
    patho_flagged  = (pred_patho  == 1).mean()

    print(f"   Normal gait correctly identified: {normal_correct*100:.1f}%")
    print(f"   Pathological gait flagged:        {patho_flagged*100:.1f}%")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(err_normal, bins=40, alpha=0.7, color="green", label="Normal Gait")
    axes[0].hist(err_patho,  bins=40, alpha=0.7, color="red",   label="Simulated Pathological")
    axes[0].axvline(threshold, color="black", linewidth=2, linestyle="--", label="Threshold")
    axes[0].set_title("Normal vs Pathological Reconstruction Error", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Reconstruction Error")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    categories = ["Normal\nGait", "Pathological\nGait (simulated)"]
    flagged    = [pred_normal.mean() * 100, pred_patho.mean() * 100]
    colors     = ["green", "red"]
    bars = axes[1].bar(categories, flagged, color=colors, alpha=0.8, width=0.4)
    for bar, val in zip(bars, flagged):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 1,
                     f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    axes[1].set_title("% Flagged as Anomaly", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("% Flagged")
    axes[1].set_ylim(0, 110)

    plt.suptitle("Pathological Gait Detection Performance",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULT_FOLDER, "pathological_detection.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📈 Detection plot saved → {path}")

    return normal_correct, patho_flagged

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GAIT PROJECT — Step 4: Anomaly Detection")
    print("=" * 60)

    df, feature_cols = load_data()

    # Use Walking data as "normal" baseline
    normal_mask = df["activity_name"].isin(NORMAL_ACTIVITIES)
    df_normal   = df[normal_mask]
    df_other    = df[~normal_mask]

    print(f"\n📊 Normal samples  : {len(df_normal):,}")
    print(f"   Other  samples  : {len(df_other):,}")

    # Create windows for normal data
    X_normal = create_windows(df_normal[feature_cols].values)
    print(f"   Normal windows  : {X_normal.shape[0]:,}")

    # Train autoencoder
    autoencoder, history = train_autoencoder(X_normal)

    # Compute threshold
    threshold, normal_errors = compute_threshold(autoencoder, X_normal)

    # Save threshold
    threshold_data = {"threshold": float(threshold)}
    with open(os.path.join(MODEL_FOLDER, "anomaly_threshold.pkl"), "wb") as f:
        pickle.dump(threshold_data, f)
    print(f"   💾 Threshold saved → models/anomaly_threshold.pkl")

    # Plot results
    plot_autoencoder_results(history, normal_errors, threshold)

    # Simulate pathological detection
    norm_acc, patho_acc = simulate_pathological_detection(
        autoencoder, X_normal, threshold
    )

    print("\n" + "=" * 60)
    print("  ✅ Anomaly Detection DONE!")
    print(f"     Anomaly Threshold   : {threshold:.6f}")
    print(f"     Normal Specificity  : {norm_acc*100:.1f}%")
    print(f"     Pathological Recall : {patho_acc*100:.1f}%")
    print("=" * 60)
    print("\n▶ Next step: python 5_evaluate.py")

if __name__ == "__main__":
    main()
