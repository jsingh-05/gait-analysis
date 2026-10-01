"""
FILE: preprocess.py
PURPOSE: Read ALL HuGaDB txt files → Clean → Save as processed CSV
RUN: python preprocess.py
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# ── SETTINGS ──────────────────────────────────────────────────────────────────
POSSIBLE_DATA_FOLDERS = ["Data", "data", "../Data", "HuGaDB", "dataset"]
OUTPUT_FOLDER = "processed"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SENSOR_COLS = [
    "RKnee_ax", "RKnee_ay", "RKnee_az", "RKnee_gx", "RKnee_gy", "RKnee_gz",
    "RHip_ax",  "RHip_ay",  "RHip_az",  "RHip_gx",  "RHip_gy",  "RHip_gz",
    "LKnee_ax", "LKnee_ay", "LKnee_az", "LKnee_gx", "LKnee_gy", "LKnee_gz",
    "LHip_ax",  "LHip_ay",  "LHip_az",  "LHip_gx",  "LHip_gy",  "LHip_gz",
    "RAcc_ax",  "RAcc_ay",  "RAcc_az",  "LAcc_ax",  "LAcc_ay",  "LAcc_az",
    "LAnkle_ax","LAnkle_ay","LAnkle_az","RAnkle_ax","RAnkle_ay","RAnkle_az",
    "EMG_R", "EMG_L",
    "activity"
]

ACTIVITY_MAP = {
    1:  "Walking",
    2:  "Running",
    3:  "Going_Up",
    4:  "Going_Down",
    5:  "Sitting",
    6:  "Sitting_Down",
    7:  "Standing_Up",
    8:  "Standing",
    9:  "Bicycling",
    10: "Up_by_Elevator",
    11: "Down_by_Elevator",
    12: "Sitting_in_Car"
}

def find_data_folder():
    for folder in POSSIBLE_DATA_FOLDERS:
        if os.path.isdir(folder):
            print(f"✅ Found data folder: '{folder}'")
            return folder
    raise FileNotFoundError(
        f"❌ Data folder not found!\nCurrent directory: {os.getcwd()}"
    )

def load_all_files(data_folder):
    all_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    if not all_files:
        raise ValueError(f"No .txt files found in '{data_folder}'!")

    print(f"\n📂 Found {len(all_files)} files. Loading...")
    all_dfs = []

    for fname in tqdm(sorted(all_files), desc="Reading files"):
        fpath = os.path.join(data_folder, fname)
        try:
            df = pd.read_csv(fpath, sep="\t", comment="#", header=None, low_memory=False)
            ncols = df.shape[1]
            if ncols == 39:
                df.columns = SENSOR_COLS
            elif ncols == 38:
                df.columns = SENSOR_COLS[:-2] + ["activity"]
            elif ncols > 39:
                df = df.iloc[:, :39]
                df.columns = SENSOR_COLS
            else:
                df = pd.read_csv(fpath, comment="#", header=None, low_memory=False)
                if df.shape[1] >= 37:
                    df = df.iloc[:, :39] if df.shape[1] >= 39 else df
                    df.columns = SENSOR_COLS[:df.shape[1]]

            df["source_file"] = fname
            parts = fname.replace(".txt", "").split("_")
            df["subject_id"] = parts[3] if len(parts) > 3 else "unknown"
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Skipping {fname}: {e}")

    if not all_dfs:
        raise ValueError("No files could be loaded!")

    raw_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n✅ Loaded {len(all_dfs)} files → {raw_df.shape[0]:,} rows × {raw_df.shape[1]} cols")
    return raw_df

def clean_data(df):
    print("\n🧹 Cleaning data...")

    feature_cols = [c for c in SENSOR_COLS[:-1] if c in df.columns]
    df_clean = df[feature_cols + ["activity"]].copy()

    before = len(df_clean)

    # Convert ALL sensor columns to numeric — fixes str/int errors
    for col in feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Convert activity to numeric
    df_clean["activity"] = pd.to_numeric(df_clean["activity"], errors="coerce")

    # Drop NaN rows created by conversion
    df_clean.dropna(inplace=True)

    # Keep only valid activity labels 1-12
    df_clean["activity"] = df_clean["activity"].astype(int)
    df_clean = df_clean[df_clean["activity"].between(1, 12)]

    # Remove duplicates
    df_clean.drop_duplicates(inplace=True)

    after = len(df_clean)
    print(f"   Rows before cleaning: {before:,}")
    print(f"   Rows after  cleaning: {after:,}")
    print(f"   Removed:              {before - after:,} rows")

    return df_clean

def add_features(df):
    print("\n⚙️  Adding engineered features...")

    accel_cols = [c for c in df.columns if "_ax" in c or "_ay" in c or "_az" in c]
    gyro_cols  = [c for c in df.columns if "_gx" in c or "_gy" in c or "_gz" in c]

    # Ensure numeric before math operations
    for col in accel_cols + gyro_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if len(accel_cols) >= 3:
        df["accel_magnitude"] = np.sqrt(
            df[accel_cols[0]]**2 + df[accel_cols[1]]**2 + df[accel_cols[2]]**2
        )

    if len(gyro_cols) >= 3:
        df["gyro_magnitude"] = np.sqrt(
            df[gyro_cols[0]]**2 + df[gyro_cols[1]]**2 + df[gyro_cols[2]]**2
        )

    r_accel = [c for c in df.columns if c.startswith("R") and "_ax" in c]
    l_accel = [c for c in df.columns if c.startswith("L") and "_ax" in c]
    if r_accel and l_accel:
        df["lr_asymmetry"] = abs(df[r_accel[0]] - df[l_accel[0]])

    print(f"   Final shape: {df.shape}")
    return df

def encode_labels(df):
    print("\n🏷️  Encoding labels...")

    df["activity_name"] = df["activity"].map(ACTIVITY_MAP)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["activity_name"])

    print(f"   Classes: {list(le.classes_)}")
    print(f"   Distribution:\n{df['activity_name'].value_counts()}")

    return df, le

def scale_features(df, feature_cols):
    print("\n📏 Scaling features...")

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    with open(os.path.join(OUTPUT_FOLDER, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("   ✅ Scaler saved to processed/scaler.pkl")
    return df, scaler

def main():
    print("=" * 60)
    print("  GAIT PROJECT — Step 1: Preprocessing")
    print("=" * 60)

    data_folder  = find_data_folder()
    raw_df       = load_all_files(data_folder)
    df           = clean_data(raw_df)
    df           = add_features(df)
    df, le       = encode_labels(df)

    skip_cols    = {"activity", "activity_name", "label", "source_file", "subject_id"}
    feature_cols = [
        c for c in df.columns
        if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    df, scaler = scale_features(df, feature_cols)

    out_path = os.path.join(OUTPUT_FOLDER, "processed_data.csv")
    df.to_csv(out_path, index=False)
    print(f"\n💾 Saved processed data  → {out_path}")

    with open(os.path.join(OUTPUT_FOLDER, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print("💾 Saved label encoder   → processed/label_encoder.pkl")

    with open(os.path.join(OUTPUT_FOLDER, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    print("💾 Saved feature columns → processed/feature_cols.pkl")

    print("\n" + "=" * 60)
    print(f"  ✅ Preprocessing DONE!")
    print(f"     Total samples : {len(df):,}")
    print(f"     Features      : {len(feature_cols)}")
    print(f"     Classes       : {len(le.classes_)}")
    print("=" * 60)
    print("\n▶ Next step: python EDA.py")

if __name__ == "__main__":
    main()