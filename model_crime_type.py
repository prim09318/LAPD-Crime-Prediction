"""
model_crime_type.py
===================
Crime Type Risk Model — Multiclass LightGBM
Task: Given area + time context → predict TOP-N most likely crime categories
Also: Victim profile risk analysis using SHAP + statistical profiling
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    top_k_accuracy_score, f1_score
)

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
for d in [MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

try:
    import cupy as cp
    LGBM_DEVICE = "gpu"
    print("✅  GPU detected — LightGBM multiclass on GPU.")
except ImportError:
    LGBM_DEVICE = "cpu"
    print("⚠️   GPU not found — using CPU.")


# ══════════════════════════════════════════════════════════════════════════════
#  PREPARE FEATURES FOR CRIME TYPE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
CRIME_TYPE_FEATURES = [
    "area", "rpt_dist_no", "hour", "month", "day_of_week", "year",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_weekend", "premis_cd", "part_1_2",
    "area_log_density", "is_property_crime",
    "time_of_day_enc",       # encoded after cat conversion
]

def prepare_crime_type_data(df: pd.DataFrame):
    """
    Target: crime_category (broad group)
    Returns X, y, label_encoder, feature_cols
    """
    print("📐  Preparing crime type classification data...")

    data = df.copy()

    # Encode time_of_day
    tod_enc = LabelEncoder()
    if "time_of_day" in data.columns:
        data["time_of_day_enc"] = tod_enc.fit_transform(
            data["time_of_day"].astype(str)
        )
    else:
        data["time_of_day_enc"] = 0

    # Fill missing premis_cd
    data["premis_cd"] = data["premis_cd"].fillna(-1).astype(int)

    # Target encoding
    le = LabelEncoder()
    data["crime_cat_enc"] = le.fit_transform(data["crime_category"].fillna("Other"))

    feature_cols = [c for c in CRIME_TYPE_FEATURES if c in data.columns]

    X = data[feature_cols].fillna(0)
    y = data["crime_cat_enc"].values

    print(f"  Classes    : {list(le.classes_)}")
    print(f"  Features   : {len(feature_cols)}")
    print(f"  Samples    : {len(X):,}\n")

    return X, y, le, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN MULTICLASS LIGHTGBM
# ══════════════════════════════════════════════════════════════════════════════
def train_crime_type_model(X: pd.DataFrame, y: np.ndarray, n_classes: int, n_folds: int = 4):
    print(f"🚀  Training multiclass LightGBM ({n_classes} classes, device={LGBM_DEVICE})...")

    params = {
        "objective":        "multiclass",
        "num_class":        n_classes,
        "metric":           "multi_logloss",
        "boosting_type":    "gbdt",
        "n_estimators":     800,
        "learning_rate":    0.05,
        "num_leaves":       63,
        "max_depth":        6,
        "min_child_samples":20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "reg_alpha":        0.1,
        "reg_lambda":       0.2,
        "class_weight":     "balanced",
        "verbose":          -1,
        "device":           LGBM_DEVICE,
        "random_state":     42,
        "n_jobs":           -1,
    }
    if LGBM_DEVICE == "gpu":
        params["gpu_platform_id"] = 0
        params["gpu_device_id"]   = 0

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    best_iters = []

    fold_bar = tqdm(enumerate(skf.split(X, y)), total=n_folds, desc="  CV Folds")

    for fold, (tr_idx, val_idx) in fold_bar:
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx],      y[val_idx]

        model = lgb.LGBMClassifier(**params)

        class TreeProgressBar:
            def __init__(self, total, fold_n):
                self.pbar = tqdm(total=total, desc=f"    Fold {fold_n+1} trees",
                                 leave=False, unit="tree")
            def __call__(self, env):
                self.pbar.update(1)
                if env.iteration == env.end_iteration - 1:
                    self.pbar.close()

        cb = TreeProgressBar(params["n_estimators"], fold)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
                cb,
            ],
        )

        proba = model.predict_proba(X_val)
        preds = np.argmax(proba, axis=1)

        f1    = f1_score(y_val, preds, average="weighted", zero_division=0)
        top3  = top_k_accuracy_score(y_val, proba, k=3)
        cv_scores.append({"fold": fold + 1, "F1_weighted": f1, "Top3_Acc": top3})
        best_iters.append(model.best_iteration_)
        fold_bar.set_postfix(F1=f"{f1:.3f}", Top3=f"{top3:.3f}")

    metrics_df = pd.DataFrame(cv_scores)
    print(f"\n  CV Results:\n{metrics_df.to_string(index=False)}")
    print(f"  Mean F1   : {metrics_df['F1_weighted'].mean():.3f} ± {metrics_df['F1_weighted'].std():.3f}")
    print(f"  Mean Top3 : {metrics_df['Top3_Acc'].mean():.3f}")

    # Final model
    print("\n  Training final model on full data...")
    final_params = {**params, "n_estimators": int(np.mean(best_iters)) + 50}
    final_model  = lgb.LGBMClassifier(**final_params)

    with tqdm(total=final_params["n_estimators"],
              desc="  Final model trees", unit="tree") as pbar:
        class FinalCB:
            def __call__(self, env): pbar.update(1)
        final_model.fit(X, y, callbacks=[lgb.log_evaluation(period=-1), FinalCB()])

    return final_model, metrics_df


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT TOP-N CRIMES FOR A GIVEN CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
def predict_top_crimes(model, le: LabelEncoder, feature_cols: list,
                       area: int, hour: int, month: int,
                       day_of_week: int, year: int,
                       premis_cd: int = -1, top_n: int = 5) -> pd.DataFrame:
    """
    Returns a DataFrame of top-N predicted crime categories with probabilities.
    """
    row = {col: 0 for col in feature_cols}
    row.update({
        "area":        area,
        "hour":        hour,
        "month":       month,
        "day_of_week": day_of_week,
        "year":        year,
        "premis_cd":   premis_cd,
        "hour_sin":    np.sin(2 * np.pi * hour  / 24),
        "hour_cos":    np.cos(2 * np.pi * hour  / 24),
        "month_sin":   np.sin(2 * np.pi * month / 12),
        "month_cos":   np.cos(2 * np.pi * month / 12),
        "dow_sin":     np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos":     np.cos(2 * np.pi * day_of_week / 7),
        "is_weekend":  int(day_of_week in [5, 6]),
    })

    X_pred = pd.DataFrame([row])[feature_cols].fillna(0)
    proba  = model.predict_proba(X_pred)[0]

    results = pd.DataFrame({
        "crime_category": le.classes_,
        "probability":    proba,
    }).sort_values("probability", ascending=False).head(top_n)
    results["rank"] = range(1, top_n + 1)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  VICTIM PROFILE ANALYSIS  (stats + SHAP)
# ══════════════════════════════════════════════════════════════════════════════
def analyse_victim_profiles(df: pd.DataFrame, save_dir: Path = OUTPUT_DIR):
    """
    Detailed victim analysis:
    - Sex breakdown per crime category
    - Descent breakdown per crime category
    - Age group distribution (known ages only)
    - Time-of-crime patterns per victim group
    - Property crime vs violent crime victim profile split
    """
    print("\n👤  Running victim profile analysis...")
    plt.style.use("dark_background")
    plots = []

    with tqdm(total=5, desc="  Victim plots") as pbar:

        # ── 1. Victim sex per crime category ─────────────────────────────────
        if "vict_sex" in df.columns and "crime_category" in df.columns:
            fig, ax = plt.subplots(figsize=(14, 6))
            sex_cat = df[df["vict_sex"].isin(["M","F"])].groupby(
                ["crime_category", "vict_sex"]
            ).size().unstack(fill_value=0)
            sex_cat.plot(kind="bar", ax=ax,
                         color=["#5C9EFF", "#E84545"], width=0.7, edgecolor="none")
            ax.set_title("Victim Sex by Crime Category", fontsize=14, fontweight="bold")
            ax.set_xlabel(""); ax.set_ylabel("Count")
            ax.legend(["Female", "Male"], loc="upper right")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
            plt.tight_layout()
            fig.savefig(save_dir / "victim_sex_by_crime.png", dpi=150)
            plt.close(fig)
        pbar.update(1)

        # ── 2. Victim descent per crime category (top 5 descents) ────────────
        if "descent_label" in df.columns:
            fig, ax = plt.subplots(figsize=(14, 7))
            top5_descents = df["descent_label"].value_counts().head(5).index
            dc = df[df["descent_label"].isin(top5_descents)].groupby(
                ["crime_category", "descent_label"]
            ).size().unstack(fill_value=0)
            dc.plot(kind="bar", ax=ax,
                    colormap="tab10", width=0.75, edgecolor="none")
            ax.set_title("Top 5 Victim Descent Groups by Crime Category",
                         fontsize=14, fontweight="bold")
            ax.set_xlabel(""); ax.set_ylabel("Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
            ax.legend(loc="upper right", fontsize=8)
            plt.tight_layout()
            fig.savefig(save_dir / "victim_descent_by_crime.png", dpi=150)
            plt.close(fig)
        pbar.update(1)

        # ── 3. Age group × crime category heatmap ────────────────────────────
        if "age_group" in df.columns:
            fig, ax = plt.subplots(figsize=(14, 7))
            age_cat = df[df["age_group"] != "Unknown"].groupby(
                ["age_group", "crime_category"]
            ).size().unstack(fill_value=0)
            # Normalise by row so colour shows proportion not raw count
            age_cat_norm = age_cat.div(age_cat.sum(axis=1), axis=0) * 100
            sns.heatmap(age_cat_norm, cmap="YlOrRd", ax=ax, annot=True, fmt=".1f",
                        linewidths=0.3, cbar_kws={"label": "% of age group"})
            ax.set_title("Crime Category Distribution by Age Group (% of row)",
                         fontsize=14, fontweight="bold")
            ax.set_xlabel("Crime Category")
            ax.set_ylabel("Victim Age Group")
            plt.tight_layout()
            fig.savefig(save_dir / "victim_age_crime_heatmap.png", dpi=150)
            plt.close(fig)
        pbar.update(1)

        # ── 4. Hour of crime by victim sex ────────────────────────────────────
        if "vict_sex" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            for sex, color, label in [("M","#5C9EFF","Male"), ("F","#E84545","Female")]:
                sub = df[df["vict_sex"] == sex].groupby("hour").size()
                sub = sub / sub.sum()  # normalise to proportion
                ax.plot(sub.index, sub.values, color=color, linewidth=2.5, label=label)
            ax.set_title("Crime Time Patterns by Victim Sex (Normalised)",
                         fontsize=14, fontweight="bold")
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Proportion of Crimes")
            ax.legend(); ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
            plt.tight_layout()
            fig.savefig(save_dir / "victim_sex_hourly.png", dpi=150)
            plt.close(fig)
        pbar.update(1)

        # ── 5. Property vs Violent crime victim profile comparison ───────────
        if "is_property_crime" in df.columns and "vict_sex" in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for idx, (flag, title) in enumerate([(0,"Violent/Personal Crimes"),
                                                  (1,"Property Crimes")]):
                sub  = df[df["is_property_crime"] == flag]
                sex_c= sub[sub["vict_sex"].isin(["M","F"])]["vict_sex"].value_counts()
                axes[idx].pie(sex_c.values, labels=["Male","Female"],
                              autopct="%1.1f%%",
                              colors=["#5C9EFF", "#E84545"],
                              startangle=90)
                axes[idx].set_title(title, fontsize=12, fontweight="bold")
            plt.suptitle("Victim Sex: Violent vs Property Crime",
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
            fig.savefig(save_dir / "victim_property_vs_violent.png", dpi=150)
            plt.close(fig)
        pbar.update(1)

    print(f"✅  Victim profile plots saved to {save_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(model, X_val, y_val, le, save_dir=OUTPUT_DIR):
    preds = model.predict(X_val)
    cm    = confusion_matrix(y_val, preds)
    cm_pct= cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title("Confusion Matrix (% of True Class)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"✅  Confusion matrix saved.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_crime_type_pipeline():
    clean_path = DATA_DIR / "clean_crime.parquet"
    assert clean_path.exists(), "Run data_pipeline.py first."

    print("📂  Loading clean data...")
    df = pd.read_parquet(clean_path)

    # ── Victim analysis (no model needed) ────────────────────────────────────
    analyse_victim_profiles(df)

    # ── Crime type classification ─────────────────────────────────────────────
    X, y, le, feature_cols = prepare_crime_type_data(df)

    n_classes = len(le.classes_)
    model, metrics_df = train_crime_type_model(X, y, n_classes)

    # Save
    joblib.dump(model,        MODEL_DIR / "crime_type_model.pkl")
    joblib.dump(le,           MODEL_DIR / "crime_type_label_encoder.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "crime_type_features.pkl")
    metrics_df.to_csv(OUTPUT_DIR / "cv_metrics_crime_type.csv", index=False)

    # Confusion matrix on a held-out 20%
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                                  random_state=42, stratify=y)
    plot_confusion_matrix(model, X_val, y_val, le)

    # SHAP
    print("\n🔍  Computing SHAP for crime type model (sample 1500)...")
    sample = X.sample(min(1500, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)  # shape: (n_classes, n_samples, n_features)
    # Plot for the most frequent class
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals[0], sample, plot_type="bar", show=False,
                      plot_size=None)
    plt.title("SHAP — Crime Type Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_crime_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n🎉  Crime type pipeline complete!\n")
    return model, le, feature_cols


if __name__ == "__main__":
    run_crime_type_pipeline()