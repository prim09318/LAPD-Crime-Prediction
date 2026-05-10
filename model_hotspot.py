"""
model_hotspot.py
================
Crime Hotspot Forecaster — LightGBM (GPU-enabled)
Task: Given Area + Week/Month + Year → predict crime COUNT per area
Output: area_intensity_scores.parquet  (feeds Streamlit heatmap)
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
for d in [MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

# ── GPU detection ──────────────────────────────────────────────────────────────
try:
    import cupy as cp
    GPU_AVAILABLE = True
    LGBM_DEVICE   = "gpu"
    print("✅  GPU detected — LightGBM will use GPU acceleration.")
except ImportError:
    GPU_AVAILABLE  = False
    LGBM_DEVICE    = "cpu"
    print("⚠️   No GPU — LightGBM on CPU (still fast for 800K rows).")


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD AGGREGATED TRAINING DATA
#  One row = (area, year, month, week) → crime_count
#  This collapses 800K incident rows into ~10K–50K aggregate rows
#  — trains in seconds, generalises better for count prediction
# ══════════════════════════════════════════════════════════════════════════════
def build_count_features(df: pd.DataFrame, granularity: str = "month") -> pd.DataFrame:
    """
    granularity: 'month' or 'week'
    Returns aggregated DataFrame with engineered lag/rolling features.
    """
    print(f"📐  Building count features (granularity={granularity})...")
    assert granularity in ("month", "week"), "granularity must be 'month' or 'week'"

    group_cols = ["area", "year", granularity]
    if granularity == "week" and "week" not in df.columns:
        raise ValueError("Column 'week' not found. Run data_pipeline first.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = (
        df.groupby(group_cols)
        .agg(
            crime_count    = ("crm_cd", "count"),
            unique_crimes  = ("crm_cd", "nunique"),
            violent_crimes = ("crime_category", lambda x: (x == "Aggravated Assault").sum()
                              if "crime_category" in df.columns else 0),
            property_crimes= ("crime_category", lambda x: (x.isin(
                ["Theft/Larceny","Vehicle Theft","Burglary"])).sum()
                              if "crime_category" in df.columns else 0),
            avg_hour       = ("hour", "mean"),
            night_crimes   = ("hour", lambda x: ((x >= 20) | (x <= 5)).sum()),
            weekend_crimes = ("is_weekend", "sum") if "is_weekend" in df.columns else ("crm_cd", "count"),
        )
        .reset_index()
    )

    # ── Sort for temporal lag features ────────────────────────────────────────
    time_col = "month" if granularity == "month" else "week"
    agg = agg.sort_values(["area", "year", time_col]).reset_index(drop=True)

    # ── Lag features (1, 2, 3 periods back per area) ─────────────────────────
    print("    Computing lag & rolling features...")
    for lag in tqdm([1, 2, 3, 6], desc="    Lags"):
        agg[f"crime_lag_{lag}"] = (
            agg.groupby("area")["crime_count"].shift(lag)
        )

    # ── Rolling mean (3-period and 6-period) ──────────────────────────────────
    for window in [3, 6]:
        agg[f"crime_roll_mean_{window}"] = (
            agg.groupby("area")["crime_count"]
               .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        agg[f"crime_roll_std_{window}"] = (
            agg.groupby("area")["crime_count"]
               .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )

    # ── YoY delta ─────────────────────────────────────────────────────────────
    periods_per_year = 12 if granularity == "month" else 52
    agg["crime_yoy_lag"] = agg.groupby("area")["crime_count"].shift(periods_per_year)

    # ── Cyclical encoding ─────────────────────────────────────────────────────
    max_period = 12 if granularity == "month" else 52
    agg[f"{time_col}_sin"] = np.sin(2 * np.pi * agg[time_col] / max_period)
    agg[f"{time_col}_cos"] = np.cos(2 * np.pi * agg[time_col] / max_period)

    # ── Area-level static features ────────────────────────────────────────────
    area_stats = (
        df.groupby("area")["crm_cd"].count().rename("area_historic_total") /
        df["year"].nunique()
    ).reset_index()
    area_stats.columns = ["area", "area_avg_annual_crimes"]
    agg = agg.merge(area_stats, on="area", how="left")

    # Drop rows with NaN lags (first few periods per area)
    agg.dropna(subset=["crime_lag_1", "crime_lag_2"], inplace=True)
    agg.reset_index(drop=True, inplace=True)

    print(f"✅  Aggregated dataset: {len(agg):,} rows\n")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN LIGHTGBM WITH TIMESERIES CV
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS_MONTH = [
    "area", "year", "month",
    "month_sin", "month_cos",
    "crime_lag_1", "crime_lag_2", "crime_lag_3", "crime_lag_6",
    "crime_roll_mean_3", "crime_roll_mean_6",
    "crime_roll_std_3", "crime_roll_std_6",
    "crime_yoy_lag",
    "unique_crimes", "violent_crimes", "property_crimes",
    "avg_hour", "night_crimes",
    "area_avg_annual_crimes",
]

FEATURE_COLS_WEEK = [c.replace("month", "week") for c in FEATURE_COLS_MONTH]


def train_hotspot_model(
    agg_df: pd.DataFrame,
    granularity: str = "month",
    n_splits: int = 4,
):
    """
    TimeSeriesSplit CV → final model trained on all data.
    Returns: (model, feature_importance_df, cv_metrics)
    """
    print(f"🚀  Training LightGBM hotspot model (device={LGBM_DEVICE})...")

    feature_cols = FEATURE_COLS_MONTH if granularity == "month" else FEATURE_COLS_WEEK
    feature_cols = [c for c in feature_cols if c in agg_df.columns]
    target_col   = "crime_count"

    X = agg_df[feature_cols].copy()
    y = agg_df[target_col].values

    # ── LightGBM params ───────────────────────────────────────────────────────
    params = {
        "objective":        "regression",
        "metric":           ["mae", "rmse"],
        "boosting_type":    "gbdt",
        "n_estimators":     1000,
        "learning_rate":    0.05,
        "num_leaves":       63,
        "max_depth":        7,
        "min_child_samples":10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "verbose":          -1,
        "device":           LGBM_DEVICE,
        "random_state":     42,
        "n_jobs":           -1,
    }
    if LGBM_DEVICE == "gpu":
        params["gpu_platform_id"] = 0
        params["gpu_device_id"]   = 0

    # ── TimeSeriesSplit CV ────────────────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_metrics = []

    print(f"\n  Running {n_splits}-fold TimeSeriesSplit CV...")
    fold_bar = tqdm(enumerate(tscv.split(X)), total=n_splits, desc="  CV Folds")

    for fold, (train_idx, val_idx) in fold_bar:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx],      y[val_idx]

        model = lgb.LGBMRegressor(**params)

        # tqdm callback for within-fold training progress
        class TQDMCallback:
            def __init__(self, total, fold_n):
                self.pbar = tqdm(total=total, desc=f"    Fold {fold_n+1} trees",
                                 leave=False, unit="tree")
            def __call__(self, env):
                self.pbar.update(1)
                if env.iteration == env.end_iteration - 1:
                    self.pbar.close()

        cb = TQDMCallback(params["n_estimators"], fold)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
                cb,
            ],
        )

        preds = model.predict(X_val)
        mae  = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2   = r2_score(y_val, preds)
        cv_metrics.append({"fold": fold + 1, "MAE": mae, "RMSE": rmse, "R2": r2})
        fold_bar.set_postfix(MAE=f"{mae:.2f}", R2=f"{r2:.3f}")

    metrics_df = pd.DataFrame(cv_metrics)
    print(f"\n  CV Results:\n{metrics_df.to_string(index=False)}")
    print(f"\n  Mean MAE : {metrics_df['MAE'].mean():.2f} ± {metrics_df['MAE'].std():.2f}")
    print(f"  Mean R²  : {metrics_df['R2'].mean():.3f} ± {metrics_df['R2'].std():.3f}")

    # ── Final model on ALL data ───────────────────────────────────────────────
    print("\n  Training final model on full dataset...")
    final_params = {**params, "n_estimators": model.best_iteration_ + 50}
    final_model = lgb.LGBMRegressor(**final_params)

    with tqdm(total=final_params["n_estimators"],
              desc="  Final model trees", unit="tree") as pbar:
        class FinalCB:
            def __call__(self, env):
                pbar.update(1)
        final_model.fit(X, y, callbacks=[lgb.log_evaluation(period=-1), FinalCB()])

    # ── Feature importance ────────────────────────────────────────────────────
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return final_model, fi_df, metrics_df, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
def compute_shap(model, X_sample: pd.DataFrame, save_dir: Path = OUTPUT_DIR):
    print("\n🔍  Computing SHAP values (sample of 2000 rows)...")
    explainer   = shap.TreeExplainer(model)
    sample      = X_sample.sample(min(2000, len(X_sample)), random_state=42)
    shap_values = explainer.shap_values(sample)

    plt.style.use("dark_background")

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False,
                      color=plt.cm.YlOrRd(0.7))
    plt.title("SHAP Feature Importance — Crime Hotspot Model",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Beeswarm
    fig2 = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, show=False)
    plt.title("SHAP Beeswarm — Feature Impact on Crime Count",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig2.savefig(save_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"✅  SHAP plots saved to {save_dir}")
    return shap_values


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT HEATMAP SCORES
#  Given user inputs (year, month/week), score every area
# ══════════════════════════════════════════════════════════════════════════════
def predict_heatmap(
    model,
    agg_df: pd.DataFrame,
    feature_cols: list,
    target_year: int,
    target_period: int,
    granularity: str = "month",
) -> pd.DataFrame:
    """
    Returns DataFrame: area | predicted_crime_count | risk_score
    risk_score is 0-1 normalised for heatmap colouring.

    Strategy: start every pred row from the area's most recent agg row
    (guarantees ALL feature_cols are present), then override only the
    time-varying fields for the requested period.  This prevents the
    feature-count mismatch that caused LightGBMError.
    """
    time_col         = "month" if granularity == "month" else "week"
    max_period       = 12     if granularity == "month" else 52
    periods_per_year = 12     if granularity == "month" else 52

    areas = agg_df["area"].unique()
    rows  = []

    for area in areas:
        area_hist = (
            agg_df[agg_df["area"] == area]
            .sort_values(["year", time_col])
            .reset_index(drop=True)
        )
        if len(area_hist) < 2:
            continue

        # ── Start from the last known row — this already has every feature ──
        pred_row = area_hist.iloc[-1].to_dict()
        counts   = area_hist["crime_count"].values

        # ── Override time-varying fields ─────────────────────────────────────
        pred_row["year"]   = target_year
        pred_row[time_col] = target_period

        pred_row[f"{time_col}_sin"] = np.sin(2 * np.pi * target_period / max_period)
        pred_row[f"{time_col}_cos"] = np.cos(2 * np.pi * target_period / max_period)

        pred_row["crime_lag_1"] = counts[-1] if len(counts) >= 1 else 0
        pred_row["crime_lag_2"] = counts[-2] if len(counts) >= 2 else 0
        pred_row["crime_lag_3"] = counts[-3] if len(counts) >= 3 else 0
        pred_row["crime_lag_6"] = counts[-6] if len(counts) >= 6 else 0

        pred_row["crime_roll_mean_3"] = (
            float(np.mean(counts[-3:])) if len(counts) >= 3 else float(np.mean(counts))
        )
        pred_row["crime_roll_mean_6"] = (
            float(np.mean(counts[-6:])) if len(counts) >= 6 else float(np.mean(counts))
        )
        pred_row["crime_roll_std_3"] = (
            float(np.std(counts[-3:])) if len(counts) >= 3 else 0.0
        )
        pred_row["crime_roll_std_6"] = (
            float(np.std(counts[-6:])) if len(counts) >= 6 else 0.0
        )
        pred_row["crime_yoy_lag"] = (
            counts[-periods_per_year] if len(counts) >= periods_per_year else counts[-1]
        )

        rows.append(pred_row)

    pred_df = pd.DataFrame(rows)

    # ── Select EXACTLY the columns the model was trained on, in the same order ─
    missing = [c for c in feature_cols if c not in pred_df.columns]
    if missing:
        print(f"⚠️  Adding zero-filled missing cols: {missing}")
        for c in missing:
            pred_df[c] = 0.0

    X_pred = pred_df[feature_cols].fillna(0).astype(float)

    # Sanity check — should never fire now
    assert X_pred.shape[1] == len(feature_cols), (
        f"Feature count mismatch: X_pred has {X_pred.shape[1]} cols, "
        f"model expects {len(feature_cols)}"
    )

    pred_df["predicted_crime_count"] = model.predict(X_pred).clip(min=0)

    mn = pred_df["predicted_crime_count"].min()
    mx = pred_df["predicted_crime_count"].max()
    pred_df["risk_score"] = (pred_df["predicted_crime_count"] - mn) / (mx - mn + 1e-9)

    return pred_df[["area", "predicted_crime_count", "risk_score"]]


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(fi_df: pd.DataFrame, save_dir: Path = OUTPUT_DIR):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 7))
    top = fi_df.head(15)
    colors = plt.cm.YlOrRd(np.linspace(0.3, 1.0, len(top)))[::-1]
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_title("LightGBM Feature Importance — Hotspot Model",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fig.savefig(save_dir / "feature_importance_hotspot.png", dpi=150)
    plt.close(fig)
    print(f"✅  Feature importance plot saved.")


# ══════════════════════════════════════════════════════════════════════════════
#  AREA CENTROIDS  (for heatmap lat/lon)
# ══════════════════════════════════════════════════════════════════════════════
def compute_area_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Compute median lat/lon per area from incident data."""
    centroids = (
        df[df["lat"].notna() & df["lon"].notna()]
        .groupby("area")
        .agg(lat=("lat", "median"), lon=("lon", "median"))
        .reset_index()
    )
    return centroids


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_hotspot_pipeline(granularity: str = "month"):
    """Full pipeline: load clean data → aggregate → train → save."""
    clean_path = DATA_DIR / "clean_crime.parquet"
    assert clean_path.exists(), "Run data_pipeline.py first."

    print("📂  Loading clean data...")
    df = pd.read_parquet(clean_path)

    # Compute centroids and save
    centroids = compute_area_centroids(df)
    centroids.to_parquet(DATA_DIR / "area_centroids.parquet", index=False)
    print(f"✅  Area centroids saved ({len(centroids)} areas)\n")

    agg_df = build_count_features(df, granularity=granularity)

    model, fi_df, metrics_df, feature_cols = train_hotspot_model(
        agg_df, granularity=granularity
    )

    # Save artefacts
    joblib.dump(model,        MODEL_DIR / f"hotspot_model_{granularity}.pkl")
    joblib.dump(feature_cols, MODEL_DIR / f"hotspot_features_{granularity}.pkl")
    agg_df.to_parquet(DATA_DIR / f"agg_{granularity}.parquet", index=False)
    metrics_df.to_csv(OUTPUT_DIR / f"cv_metrics_{granularity}.csv", index=False)

    print(f"\n💾  Model saved to {MODEL_DIR / f'hotspot_model_{granularity}.pkl'}")

    plot_feature_importance(fi_df)

    # SHAP on a sample
    X_all = agg_df[[c for c in feature_cols if c in agg_df.columns]]
    compute_shap(model, X_all)

    print("\n🎉  Hotspot pipeline complete!\n")
    return model, agg_df, feature_cols


if __name__ == "__main__":
    import sys
    gran = sys.argv[1] if len(sys.argv) > 1 else "month"
    run_hotspot_pipeline(gran)
