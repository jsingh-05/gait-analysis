"""
FILE 2: 2_eda.py
PURPOSE: Exploratory Data Analysis — graphs, correlation matrix, class distribution
RUN: python 2_eda.py   (run AFTER 1_preprocess.py)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── SETTINGS ──────────────────────────────────────────────────────────────────
INPUT_FILE  = os.path.join("processed", "processed_data.csv")
EDA_FOLDER  = os.path.join("result", "eda_graphs")
os.makedirs(EDA_FOLDER, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = "tab10"
DPI = 150

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"❌ '{INPUT_FILE}' not found! Run 1_preprocess.py first."
        )
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded data: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df

# ── PLOT 1: Class Distribution ────────────────────────────────────────────────
def plot_class_distribution(df):
    print("\n📊 Plot 1: Class Distribution")
    counts = df["activity_name"].value_counts().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    sns.barplot(x=counts.values, y=counts.index, ax=axes[0], palette=PALETTE)
    axes[0].set_title("Activity Count (Bar)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Samples")
    axes[0].set_ylabel("Activity")
    for i, v in enumerate(counts.values):
        axes[0].text(v + 100, i, f"{v:,}", va="center", fontsize=9)

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                startangle=140, colors=sns.color_palette(PALETTE, len(counts)))
    axes[1].set_title("Activity Distribution (%)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(EDA_FOLDER, "01_class_distribution.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {path}")

# ── PLOT 2: Sensor Signal Over Time (per activity) ────────────────────────────
def plot_signal_samples(df):
    print("\n📊 Plot 2: Sensor Signal Samples")

    activities = df["activity_name"].unique()[:6]  # show 6 activities
    sensor = "RKnee_ax" if "RKnee_ax" in df.columns else df.columns[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, act in enumerate(activities):
        subset = df[df["activity_name"] == act][sensor].values[:300]
        axes[i].plot(subset, color=sns.color_palette(PALETTE)[i], linewidth=0.8)
        axes[i].set_title(f"{act}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Sample Index")
        axes[i].set_ylabel(sensor)
        axes[i].set_ylim(-5, 5)

    plt.suptitle("Sensor Signal Patterns per Activity", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(EDA_FOLDER, "02_signal_per_activity.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {path}")

# ── PLOT 3: Correlation Matrix ────────────────────────────────────────────────
def plot_correlation_matrix(df):
    print("\n📊 Plot 3: Correlation Matrix")

    skip = {"activity", "activity_name", "label", "source_file", "subject_id"}
    num_cols = [c for c in df.columns if c not in skip and df[c].dtype in [float, np.float64, int, np.int64]]
    num_cols = num_cols[:20]  # top 20 features for readability

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask
    sns.heatmap(corr, mask=mask, annot=False, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix (Lower Triangle)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    path = os.path.join(EDA_FOLDER, "03_correlation_matrix.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {path}")

# ── PLOT 4: Feature Distributions ─────────────────────────────────────────────
def plot_feature_distributions(df):
    print("\n📊 Plot 4: Feature Distributions")

    key_features = ["RKnee_ax", "RKnee_gx", "LKnee_ax", "LKnee_gx",
                    "accel_magnitude", "gyro_magnitude"]
    key_features = [f for f in key_features if f in df.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        for act in df["activity_name"].unique():
            vals = df[df["activity_name"] == act][feat].dropna()
            axes[i].hist(vals, bins=50, alpha=0.4, label=act, density=True)
        axes[i].set_title(feat, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")

    axes[-1].legend(loc="upper right", fontsize=7, ncol=2)
    plt.suptitle("Feature Value Distributions by Activity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(EDA_FOLDER, "04_feature_distributions.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {path}")

# ── PLOT 5: Box Plots per Activity ────────────────────────────────────────────
def plot_boxplots(df):
    print("\n📊 Plot 5: Box Plots")

    key_features = ["RKnee_ax", "LKnee_ax", "accel_magnitude", "lr_asymmetry"]
    key_features = [f for f in key_features if f in df.columns][:4]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        sns.boxplot(data=df, x="activity_name", y=feat,
                    ax=axes[i], palette=PALETTE)
        axes[i].set_title(f"{feat} by Activity", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel(feat)
        axes[i].tick_params(axis="x", rotation=40)

    plt.suptitle("Feature Box Plots Across Activities", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(EDA_FOLDER, "05_boxplots.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {path}")

# ── PLOT 6: Missing Values Heatmap ────────────────────────────────────────────
def plot_missing_values(df):
    print("\n📊 Plot 6: Missing Values")

    skip = {"activity_name", "source_file", "subject_id"}
    check_cols = [c for c in df.columns if c not in skip][:30]

    missing = df[check_cols].isnull().sum()
    missing_pct = (missing / len(df)) * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["green" if v == 0 else "red" for v in missing_pct.values]
    bars = ax.bar(missing_pct.index, missing_pct.values, color=colors)
    ax.set_title("Missing Values (%) per Feature", fontsize=13, fontweight="bold")
    ax.set_ylabel("Missing %")
    ax.set_xlabel("Features")
    plt.xticks(rotation=45, ha="right", fontsize=8)

    # Add value labels on bars
    for bar, val in zip(bars, missing_pct.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                    f"{val:.1f}%", ha="center", fontsize=8)

    ax.text(0.98, 0.95, "Green = No Missing | Red = Has Missing",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(EDA_FOLDER, "06_missing_values.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {path}")

# ── PLOT 7: Subject-wise Activity Distribution ─────────────────────────────────
def plot_subject_distribution(df):
    print("\n📊 Plot 7: Subject-wise Distribution")

    if "subject_id" not in df.columns:
        print("   Skipped (no subject_id column)")
        return

    pivot = df.groupby(["subject_id", "activity_name"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap=PALETTE, width=0.8)
    ax.set_title("Activity Distribution per Subject", fontsize=13, fontweight="bold")
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Sample Count")
    ax.legend(loc="upper right", fontsize=8, ncol=2, title="Activity")
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = os.path.join(EDA_FOLDER, "07_subject_distribution.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {path}")

# ── PLOT 8: Pairplot for Key Features ────────────────────────────────────────
def plot_pairplot(df):
    print("\n📊 Plot 8: Pairplot")

    key_features = ["RKnee_ax", "RKnee_gx", "accel_magnitude", "activity_name"]
    key_features = [f for f in key_features if f in df.columns]

    if len(key_features) < 3:
        print("   Skipped (not enough features)")
        return

    # Sample for speed
    sample = df.sample(min(3000, len(df)), random_state=42)

    pair = sns.pairplot(sample[key_features], hue="activity_name",
                        palette=PALETTE, diag_kind="kde",
                        plot_kws={"alpha": 0.3, "s": 10})
    pair.fig.suptitle("Pairplot of Key Features by Activity", y=1.02,
                      fontsize=13, fontweight="bold")
    path = os.path.join(EDA_FOLDER, "08_pairplot.png")
    pair.fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {path}")

# ── PLOT 9: Statistical Summary Table ────────────────────────────────────────
def plot_stats_summary(df):
    print("\n📊 Plot 9: Statistical Summary")

    skip = {"activity", "activity_name", "label", "source_file", "subject_id"}
    num_cols = [c for c in df.columns if c not in skip and df[c].dtype in [float, np.float64, int, np.int64]]
    num_cols = num_cols[:10]

    stats = df[num_cols].describe().T[["mean", "std", "min", "max"]]
    stats = stats.round(3)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    table = ax.table(
        cellText=stats.values,
        rowLabels=stats.index,
        colLabels=["Mean", "Std Dev", "Min", "Max"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Color header
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#D9E1F2")

    ax.set_title("Statistical Summary of Sensor Features",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(EDA_FOLDER, "09_stats_summary.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {path}")

# ── PLOT 10: Asymmetry Analysis (Key for Pathological Detection) ───────────────
def plot_asymmetry(df):
    print("\n📊 Plot 10: Asymmetry Analysis")

    if "lr_asymmetry" not in df.columns:
        print("   Skipped (no lr_asymmetry column)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Violin plot
    sns.violinplot(data=df, x="activity_name", y="lr_asymmetry",
                   ax=axes[0], palette=PALETTE, inner="quartile")
    axes[0].set_title("Left-Right Asymmetry per Activity\n(Higher = More Abnormal)",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].set_ylabel("LR Asymmetry Score")

    # Histogram
    for act in df["activity_name"].unique():
        vals = df[df["activity_name"] == act]["lr_asymmetry"]
        axes[1].hist(vals, bins=40, alpha=0.5, label=act, density=True)
    axes[1].set_title("Asymmetry Distribution by Activity", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Asymmetry Value")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=7, ncol=2)

    plt.suptitle("Gait Asymmetry Analysis (Used for Pathological Detection)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(EDA_FOLDER, "10_asymmetry_analysis.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GAIT PROJECT — Step 2: Exploratory Data Analysis")
    print("=" * 60)

    df = load_data()

    # Run all plots
    plot_class_distribution(df)
    plot_signal_samples(df)
    plot_correlation_matrix(df)
    plot_feature_distributions(df)
    plot_boxplots(df)
    plot_missing_values(df)
    plot_subject_distribution(df)
    plot_pairplot(df)
    plot_stats_summary(df)
    plot_asymmetry(df)

    print("\n" + "=" * 60)
    print(f"  ✅ EDA DONE! All 10 graphs saved in: {EDA_FOLDER}/")
    print("=" * 60)
    print("\n▶ Next step: python 3_train_classifier.py")

if __name__ == "__main__":
    main()
