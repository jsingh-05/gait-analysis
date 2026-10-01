"""
FILE 6: 6_predict.py
PURPOSE: Final demo — take sensor data → predict Activity + Gait Status
RUN: python 6_predict.py
"""

import os
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ── SETTINGS ──────────────────────────────────────────────────────────────────
MODEL_FOLDER  = "models"
PROCESSED_DIR = "processed"
WINDOW_SIZE   = 128

ACTIVITY_DESC = {
    "Walking": "Normal walking on flat surface",
    "Running": "Running at various speeds",
    "Going_Up": "Climbing stairs upward",
    "Going_Down": "Climbing stairs downward",
    "Sitting": "Sitting on a chair",
    "Sitting_Down": "Action of sitting down",
    "Standing_Up": "Action of standing up",
    "Standing": "Static standing",
    "Bicycling": "Riding a bicycle",
    "Up_by_Elevator": "Elevator going up",
    "Down_by_Elevator": "Elevator going down",
    "Sitting_in_Car": "Sitting in a car"
}

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
def load_all():
    print("📂 Loading models...", end=" ")

    with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(PROCESSED_DIR, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)

    classifier = None
    for name in ["cnn_bilstm_best.keras", "transformer_best.keras"]:
        path = os.path.join(MODEL_FOLDER, name)
        if os.path.exists(path):
            classifier = tf.keras.models.load_model(path)
            clf_name = name
            break

    if classifier is None:
        raise FileNotFoundError("❌ No classifier found!")

    ae_path = os.path.join(MODEL_FOLDER, "autoencoder_best.keras")
    autoencoder = tf.keras.models.load_model(ae_path) if os.path.exists(ae_path) else None

    threshold = None
    th_path = os.path.join(MODEL_FOLDER, "anomaly_threshold.pkl")
    if os.path.exists(th_path):
        with open(th_path, "rb") as f:
            threshold = pickle.load(f)["threshold"]

    print(f"✅ ({clf_name})")
    return classifier, autoencoder, le, scaler, feature_cols, threshold

# ── PREDICT FUNCTION ──────────────────────────────────────────────────────────
def predict_gait(sensor_window, classifier, autoencoder, le, scaler,
                 feature_cols, threshold):

    window_scaled = scaler.transform(sensor_window)
    X = window_scaled[np.newaxis, :, :]

    # Classification
    proba = classifier.predict(X, verbose=0)[0]
    label_idx = np.argmax(proba)
    confidence = proba[label_idx]
    activity = le.inverse_transform([label_idx])[0]

    # Anomaly detection
    gait_status = "NORMAL ✅"
    recon_error = None

    if autoencoder is not None and threshold is not None:
        X_recon = autoencoder.predict(X, verbose=0)
        recon_error = float(np.mean(np.square(X - X_recon)))
        if recon_error > threshold:
            gait_status = "ABNORMAL ⚠️"

    return activity, confidence, gait_status, recon_error

# ── DISPLAY ───────────────────────────────────────────────────────────────────
def display_output(activity, confidence, gait_status, recon_error):
    condition = "Healthy Gait"
    if "ABNORMAL" in gait_status:
        condition = "Possible Mobility Issue / Abnormal Gait Pattern"
        if activity == "Walking":
            condition = "Possible Hemiplegic Limp or Mobility Disorder"

    print("\n" + "═" * 50)
    print("  🦵  GAIT ANALYSIS RESULT")
    print("═" * 50)
    print(f"  Activity     : {activity}")
    print(f"  Description  : {ACTIVITY_DESC.get(activity, activity)}")
    print(f"  Confidence   : {confidence*100:.1f}%")
    print(f"  Gait Status  : {gait_status}")
    if recon_error is not None:
        print(f"  Recon Error  : {recon_error:.6f}")
    print(f"  Condition    : {condition}")
    print("═" * 50)

# ── REAL DATA DEMO (FIXED) ────────────────────────────────────────────────────
def run_real_demo(classifier, autoencoder, le, scaler, feature_cols, threshold):
    print("\n🧪 DEMO: Real Data Sample")

    X_test_path = os.path.join(MODEL_FOLDER, "X_test.npy")
    y_test_path = os.path.join(MODEL_FOLDER, "y_test.npy")

    if not os.path.exists(X_test_path):
        print("❌ No test data found!")
        return

    X_test = np.load(X_test_path)

    idx = np.random.randint(0, len(X_test))
    window = X_test[idx]

    # already scaled → don't scale again
    X = window[np.newaxis, :, :]

    proba = classifier.predict(X, verbose=0)[0]
    label_idx = np.argmax(proba)
    confidence = proba[label_idx]
    activity = le.inverse_transform([label_idx])[0]

    gait_status = "NORMAL ✅"
    recon_error = None

    if autoencoder and threshold:
        X_recon = autoencoder.predict(X, verbose=0)
        recon_error = float(np.mean(np.square(X - X_recon)))
        if recon_error > threshold:
            gait_status = "ABNORMAL ⚠️"

    display_output(activity, confidence, gait_status, recon_error)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GAIT PROJECT — Step 6: Prediction Demo")
    print("=" * 60)

    classifier, autoencoder, le, scaler, feature_cols, threshold = load_all()

    # ✅ FIXED DEMO
    run_real_demo(classifier, autoencoder, le, scaler, feature_cols, threshold)

    print("\n" + "=" * 60)
    print("  ✅ Prediction demo COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
