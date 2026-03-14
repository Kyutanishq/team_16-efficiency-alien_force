"""
Token Delisting Risk Score Model
=================================
CoinDCX Internal Hackathon 2025 — Build for Efficiency

Pipeline:
  1. Load dataset
  2. Engineer volume ratio features (compress magnitude via ratios)
  3. Correlation analysis — drop redundant volume features (>0.85 inter-corr)
  4. Build final feature matrix (ratios + price + Santiment dev_activity)
  5. Method 1 — Stratified 5-Fold Cross-Validation (time-series proxy)
  6. Method 2 — Random Stratified 80/20 Split
  7. Compare 4 models: XGBoost, Random Forest, RF High-Precision, Logistic Regression
  8. Score all tokens → risk_score (0–100) + risk_tier (High / Medium / Low)
  9. Output: token_risk_scores.csv, model_metrics.csv, model_results.png

Usage:
  python token_delisting_risk_model.py --input your_data.csv --output ./results

Requirements:
  pip install pandas numpy scipy scikit-learn matplotlib
"""

import argparse
import os
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT  = "hackathon___overall_data_creation__1_.csv"
DEFAULT_OUTPUT = "./results"

# Columns to exclude from features
DROP_COLS = ["target_token", "status", "delisted_date", "week_", "slug"]

# Max null % for a column to be included in modelling
NULL_THRESHOLD = 0.75

# Correlation threshold above which we drop redundant volume features
CORR_DROP_THRESHOLD = 0.85

# Min precision target when finding the high-precision operating threshold
MIN_PRECISION_TARGET = 0.55

# Risk score tier boundaries (0–100)
TIER_BINS   = [0, 30, 60, 100]
TIER_LABELS = ["Low", "Medium", "High"]

# CV folds
N_FOLDS = 5

# Random seed
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"STEP 1 — Loading data: {path}")
    print(f"{'='*60}")
    df = pd.read_csv(path)
    print(f"  Shape         : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Unique tokens : {df['target_token'].nunique():,}")
    print(f"  Unique weeks  : {df['week_'].nunique()}")
    print(f"  Status counts : {df['status'].value_counts().to_dict()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ENGINEER VOLUME RATIO FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_volume_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert absolute USDT volume columns into ratios.
    Ratios (shorter window / longer window) naturally normalise magnitude
    and capture momentum direction without needing log transforms.
    
    Naming convention:
      g_ = global, d_ = DCX
      spot/fut = spot or futures
      Nw_Mw = N-week / M-week (e.g. 4w_13w = recent 4-week vs 13-week baseline)
    """
    print(f"\n{'='*60}")
    print("STEP 2 — Engineering volume ratio features")
    print(f"{'='*60}")

    r = pd.DataFrame(index=df.index)

    # ── Global spot momentum ratios ──────────────────────────────────────────
    r["g_spot_4w_8w"]   = df["global_spot_vol_4w_sum"]  / (df["global_spot_vol_8w_sum"]  + 1)
    r["g_spot_8w_13w"]  = df["global_spot_vol_8w_sum"]  / (df["global_spot_vol_13w_sum"] + 1)
    r["g_spot_1w_4w"]   = df["global_spot_vol_usdt"]    / (df["global_spot_vol_4w_sum"]  / 4 + 1)

    # ── DCX spot momentum ratios ─────────────────────────────────────────────
    r["d_spot_4w_13w"]  = df["dcx_spot_vol_4w_sum"]     / (df["dcx_spot_vol_13w_sum"]    + 1)
    r["d_spot_8w_13w"]  = df["dcx_spot_vol_8w_sum"]     / (df["dcx_spot_vol_13w_sum"]    + 1)
    r["d_spot_1w_4w"]   = df["dcx_spot_vol_usdt"]       / (df["dcx_spot_vol_4w_sum"]     / 4 + 1)

    # ── Global futures momentum ratios ───────────────────────────────────────
    r["g_fut_8w_13w"]   = df["global_futures_vol_8w_sum"] / (df["global_futures_vol_13w_sum"] + 1)
    r["d_fut_2w_8w"]    = df["dcx_futures_vol_2w_sum"]    / (df["dcx_futures_vol_8w_sum"]     + 1)

    # ── Market share ratios: DCX vs Global ───────────────────────────────────
    r["dcx_vs_g_spot_13w_ratio"] = df["dcx_spot_vol_13w_sum"]   / (df["global_spot_vol_13w_sum"]   + 1)
    r["dcx_vs_g_fut_4w_ratio"]   = df["dcx_futures_vol_4w_sum"] / (df["global_futures_vol_4w_sum"] + 1)

    # ── Futures vs Spot health ───────────────────────────────────────────────
    r["fut_vs_spot_global"]      = df["global_futures_vol_4w_sum"] / (df["global_spot_vol_4w_sum"] + 1)
    r["fut_vs_spot_dcx"]         = df["dcx_futures_vol_4w_sum"]    / (df["dcx_spot_vol_4w_sum"]    + 1)

    # ── Additional raw-computed ratios from original dataset ─────────────────
    # (already normalised — keep as-is)
    r["g_spot_4w_13w"]           = df["global_spot_vol_4w_sum"]  / (df["global_spot_vol_13w_sum"] + 1)
    r["g_fut_4w_13w"]            = df["global_futures_vol_4w_sum"] / (df["global_futures_vol_13w_sum"] + 1)
    r["g_fut_2w_8w"]             = df["global_futures_vol_2w_sum"] / (df["global_futures_vol_8w_sum"]  + 1)
    r["d_spot_4w_8w"]            = df["dcx_spot_vol_4w_sum"]     / (df["dcx_spot_vol_8w_sum"]     + 1)
    r["d_fut_4w_13w"]            = df["dcx_futures_vol_4w_sum"]  / (df["dcx_futures_vol_13w_sum"] + 1)
    r["dcx_vs_g_spot_ratio"]     = df["dcx_spot_vol_usdt"]       / (df["global_spot_vol_usdt"]    + 1)
    r["dcx_vs_g_spot_4w_ratio"]  = df["dcx_spot_vol_4w_sum"]     / (df["global_spot_vol_4w_sum"]  + 1)

    # ── Clean: replace inf, winsorise at 1st/99th percentile ────────────────
    r = r.replace([np.inf, -np.inf], np.nan)
    for col in r.columns:
        r[col] = r[col].fillna(r[col].median())
        lo, hi = r[col].quantile([0.01, 0.99])
        r[col] = r[col].clip(lo, hi)

    print(f"  Created {len(r.columns)} volume ratio features")
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CORRELATION ANALYSIS & FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def select_volume_features(
    vol_ratios: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = CORR_DROP_THRESHOLD,
) -> list:
    """
    1. Compute point-biserial correlation of each ratio with the delisting label.
    2. Drop features with inter-feature correlation > corr_threshold,
       keeping the one with higher absolute correlation to the label.
    Returns list of selected column names.
    """
    print(f"\n{'='*60}")
    print("STEP 3 — Correlation analysis & redundancy removal")
    print(f"{'='*60}")

    # Correlation with target
    corrs = {}
    for col in vol_ratios.columns:
        r, p = stats.pointbiserialr(y, vol_ratios[col])
        corrs[col] = abs(r)

    print(f"  All volume ratio correlations with label:")
    for col, r in sorted(corrs.items(), key=lambda x: x[1], reverse=True):
        r_signed, _ = stats.pointbiserialr(y, vol_ratios[col])
        print(f"    {col:<40} r={r_signed:+.4f}")

    # Inter-feature correlation matrix
    corr_matrix = vol_ratios.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper.columns:
        high_corr_partners = upper[col][upper[col] > corr_threshold].index.tolist()
        for partner in high_corr_partners:
            # Keep whichever has higher abs correlation with the label
            if corrs.get(col, 0) >= corrs.get(partner, 0):
                to_drop.add(partner)
            else:
                to_drop.add(col)

    selected = [c for c in vol_ratios.columns if c not in to_drop]
    print(f"\n  Dropped {len(to_drop)} redundant features (inter-corr > {corr_threshold}): {to_drop}")
    print(f"  Selected {len(selected)} volume features: {selected}")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD FINAL FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(df: pd.DataFrame, vol_ratios: pd.DataFrame, selected_ratios: list) -> pd.DataFrame:
    """
    Combine:
      - Selected volume ratio features (already winsorised)
      - Direct normalised features from original dataset (percentile ranks, pct_change)
      - Price features (RobustScaler applied)
      - Santiment dev_activity features (partial coverage, ~22% non-null)
        with a binary missing-indicator flag added per column
    """
    print(f"\n{'='*60}")
    print("STEP 4 — Building final feature matrix")
    print(f"{'='*60}")

    parts = [vol_ratios[selected_ratios]]

    # ── Already-normalised direct features ───────────────────────────────────
    direct_cols = [
        "dcx_spot_vol_percentile_rank", "global_spot_vol_percentile_rank",
        "dcx_futures_vol_percentile_rank", "global_futures_vol_percentile_rank",
        "dcx_spot_vol_4w_sum_percentile_rank", "dcx_spot_vol_13w_sum_percentile_rank",
        "dead_weeks_last_4w", "dcx_spot_share_pct_change_vs_13w_avg",
        "global_spot_vol_pct_change_4w", "global_spot_vol_pct_change_13w",
        "dcx_spot_vol_pct_change_4w", "dcx_spot_vol_pct_change_13w",
        "dcx_futures_to_spot_ratio_pct_change_vs_13w_avg",
    ]
    direct_cols = [c for c in direct_cols if c in df.columns]
    direct_df = df[direct_cols].copy()
    for col in direct_df.columns:
        direct_df[col] = pd.to_numeric(direct_df[col], errors="coerce")
        direct_df[col] = direct_df[col].fillna(direct_df[col].median()).fillna(0)
    direct_df = direct_df.replace([np.inf, -np.inf], 0)
    parts.append(direct_df)

    # ── Price features — RobustScaler ────────────────────────────────────────
    price_cols = [
        "price_vs_ma30_pct", "price_vs_ma90_pct", "drawdown_from_90d_peak_pct",
        "red_weeks_of_last_8", "token_return_30d_pct", "btc_divergence_pct",
        "volatility_7d_cv", "volatility_30d_cv", "volatility_regime_ratio",
        "momentum_7d_pct", "price_acceleration", "price_position_30d",
        "price_return_7d_pct", "price_return_30d_pct", "hl_spread_ratio",
        "candle_body_ratio", "range_compression_ratio", "oc_diff_vs_avg_ratio",
    ]
    price_cols = [c for c in price_cols if c in df.columns]
    price_df = df[price_cols].copy()
    for col in price_df.columns:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce").fillna(0)
    price_df = price_df.replace([np.inf, -np.inf], 0)
    scaler = RobustScaler()
    price_df[price_cols] = scaler.fit_transform(price_df)
    parts.append(price_df)

    # ── Santiment dev_activity (partial, ~22% non-null) ───────────────────────
    # Add a binary missing-indicator flag for each — the absence of Santiment
    # data is itself a signal (obscure/low-quality tokens often have no coverage)
    santiment_cols = [
        "dev_activity_1d", "dev_activity_change_7d",
        "dev_activity_change_30d", "30d_moving_avg_dev_activity_change_1d",
    ]
    santiment_cols = [c for c in santiment_cols if c in df.columns]
    for col in santiment_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        flag = s.isnull().astype(int).rename(f"{col}_is_missing")
        s = s.fillna(s.median()).fillna(0)
        parts.append(s.to_frame())
        parts.append(flag.to_frame())

    X = pd.concat(parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Final feature matrix shape : {X.shape}")
    print(f"  Feature groups:")
    print(f"    Volume ratios (selected) : {len(selected_ratios)}")
    print(f"    Direct normalised        : {len(direct_cols)}")
    print(f"    Price (RobustScaled)     : {len(price_cols)}")
    print(f"    Santiment + flags        : {len(santiment_cols)*2}")
    return X


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def find_high_precision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = MIN_PRECISION_TARGET,
) -> float:
    """
    Find the operating threshold that maximises F1 subject to Precision >= min_precision.
    Falls back to F1-optimal threshold if no threshold achieves min_precision.
    """
    prec, rec, thrs = precision_recall_curve(y_true, y_prob)
    candidates = [
        (t, p, r)
        for t, p, r in zip(thrs, prec[:-1], rec[:-1])
        if p >= min_precision and r > 0
    ]
    if candidates:
        best = max(candidates, key=lambda x: 2 * x[1] * x[2] / (x[1] + x[2] + 1e-9))
        return round(float(best[0]), 3)
    # Fallback: F1-optimal
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    return round(float(thrs[np.argmax(f1s[:-1])]), 3)


def get_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob), 4),
        "PR-AUC":    round(average_precision_score(y_true, y_prob), 4),
        "Threshold": threshold,
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Flagged":   int(y_pred.sum()),
        "TP":        int(((y_pred == 1) & (np.array(y_true) == 1)).sum()),
        "FP":        int(((y_pred == 1) & (np.array(y_true) == 0)).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_models() -> dict:
    """
    Four models covering the precision-recall trade-off spectrum:

    XGBoost (GradientBoostingClassifier):
      - Best for tabular data with mixed feature types
      - No explicit class_weight; relies on threshold tuning for precision
      - subsample + max_features for regularisation

    Random Forest (balanced):
      - class_weight='balanced' upweights minority class automatically
      - Robust to outliers, low variance

    RF High Precision:
      - Larger min_samples_leaf = more conservative splits = higher precision
      - Manual class_weight {0:1, 1:4} — less aggressive than 'balanced'

    Logistic Regression:
      - Linear baseline — useful for interpretability
      - class_weight {0:1, 1:8} matches approximate imbalance ratio
      - Low C (=0.05) = strong L2 regularisation to prevent overfit
    """
    return {
        "XGBoost": GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.75,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "RF High Prec": RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=15,
            class_weight={0: 1, 1: 4},
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "Logistic Reg": LogisticRegression(
            C=0.05,
            penalty="l2",
            class_weight={0: 1, 1: 8},
            solver="lbfgs",
            max_iter=2000,
            random_state=SEED,
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — METHOD 1: STRATIFIED K-FOLD CV
# ═══════════════════════════════════════════════════════════════════════════════

def run_method1_cv(X: pd.DataFrame, y: pd.Series, models: dict) -> tuple:
    """
    Stratified 5-Fold Cross-Validation.

    Why CV instead of time-split on single-week data:
      This dataset is a cross-sectional snapshot (1 week). CV is the appropriate
      evaluation strategy. For a full multi-week panel, replace with:
        train = df[df['week_'] < cutoff_week]
        test  = df[df['week_'] >= cutoff_week]

    Returns:
      cv_results: dict of metrics per model
      cv_oofs:    dict of out-of-fold probability arrays per model
    """
    print(f"\n{'='*60}")
    print(f"STEP 5 — Method 1: {N_FOLDS}-Fold Stratified CV")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_results, cv_oofs = {}, {}

    for name, model in models.items():
        oof = np.zeros(len(y))
        fold_aucs = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            m = type(model)(**model.get_params())
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            probs = m.predict_proba(X.iloc[va_idx])[:, 1]
            oof[va_idx] = probs
            fold_auc = roc_auc_score(y.iloc[va_idx], probs)
            fold_aucs.append(fold_auc)
            print(f"  {name:<20} Fold {fold+1}: AUC={fold_auc:.4f}")

        thr = find_high_precision_threshold(y, oof)
        metrics = get_metrics(y, oof, thr)
        metrics["CV-AUC-mean"] = round(np.mean(fold_aucs), 4)
        metrics["CV-AUC-std"]  = round(np.std(fold_aucs), 4)

        cv_results[name] = metrics
        cv_oofs[name]    = oof

        print(f"\n  {name} — OOF summary:")
        print(f"    CV AUC   : {metrics['CV-AUC-mean']} ± {metrics['CV-AUC-std']}")
        print(f"    Threshold: {thr} (high-precision target ≥{MIN_PRECISION_TARGET})")
        print(f"    Precision: {metrics['Precision']}  Recall: {metrics['Recall']}  F1: {metrics['F1']}")
        print(f"    Flagged  : {metrics['Flagged']}  TP: {metrics['TP']}  FP: {metrics['FP']}\n")

    return cv_results, cv_oofs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — METHOD 2: RANDOM STRATIFIED SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def run_method2_random(X: pd.DataFrame, y: pd.Series, models: dict) -> tuple:
    """
    Random stratified 80/20 train/test split.

    Note: For time-series panel data this can cause leakage (a token's week-15
    row in train while week-14 is in test). Use for upper-bound benchmarking only.
    Always use Method 1 as your primary reported metric.

    Returns:
      rand_results: dict of metrics per model
      rand_fitted:  dict of fitted model objects (for inspection)
      X_tr, X_te, y_tr, y_te: train/test splits
    """
    print(f"\n{'='*60}")
    print("STEP 6 — Method 2: Random Stratified 80/20 Split")
    print(f"{'='*60}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"  Train: {len(X_tr):,} rows | Test: {len(X_te):,} rows")
    print(f"  Test positives (delisted): {y_te.sum()} | negatives: {(y_te==0).sum()}")

    rand_results, rand_fitted = {}, {}

    for name, model in models.items():
        m = type(model)(**model.get_params())
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_te)[:, 1]
        thr   = find_high_precision_threshold(y_te, probs)
        metrics = get_metrics(y_te, probs, thr)

        rand_results[name] = {**metrics, "probs": probs, "y_te": y_te}
        rand_fitted[name]  = m

        print(f"\n  {name}:")
        print(f"    Threshold: {thr}  Precision: {metrics['Precision']}  "
              f"Recall: {metrics['Recall']}  F1: {metrics['F1']}")
        print(f"    ROC-AUC: {metrics['ROC-AUC']}  PR-AUC: {metrics['PR-AUC']}")
        print(f"    Flagged: {metrics['Flagged']}  TP: {metrics['TP']}  FP: {metrics['FP']}")

    return rand_results, rand_fitted, X_tr, X_te, y_tr, y_te


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — FINAL SCORING (all tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def score_all_tokens(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    model,
    model_name: str = "XGBoost",
) -> pd.DataFrame:
    """
    Retrain the best model on the full dataset and score every token.
    Returns a DataFrame with risk_score (0–100), risk_tier, top_drivers.
    """
    print(f"\n{'='*60}")
    print(f"STEP 7 — Final scoring with {model_name} (all tokens)")
    print(f"{'='*60}")

    model.fit(X, y)
    all_probs = model.predict_proba(X)[:, 1]

    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_fi_cols = fi.head(20).index.tolist()

    def top3_drivers(i: int) -> str:
        row = X.iloc[i]
        contribs = [(feat, abs(row[feat]) * fi.get(feat, 0)) for feat in top_fi_cols if feat in row.index]
        contribs.sort(key=lambda x: x[1], reverse=True)
        return " | ".join([c[0] for c in contribs[:3]])

    risk_df = pd.DataFrame({
        "token":            df["target_token"].values,
        "week":             df["week_"].values,
        "actual_status":    df["status"].values,
        "delisted_date":    df["delisted_date"].values,
        "actual_label":     y.values,
        "risk_score":       (all_probs * 100).round(1),
        "risk_probability": all_probs.round(4),
        "risk_tier":        pd.cut(
            all_probs * 100,
            bins=TIER_BINS,
            labels=TIER_LABELS,
            include_lowest=True,
        ),
        "top_drivers":      [top3_drivers(i) for i in range(len(X))],
    })
    risk_df = risk_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    print(f"\n  Risk Tier Breakdown:")
    for tier in ["High", "Medium", "Low"]:
        sub  = risk_df[risk_df["risk_tier"] == tier]
        n_d  = (sub["actual_status"] == "delisted").sum()
        prec = n_d / max(len(sub), 1) * 100
        rec  = n_d / max(y.sum(), 1) * 100
        print(f"    {tier:6}: {len(sub):5} tokens | {n_d:4} delisted | "
              f"Precision={prec:.1f}% | Recall={rec:.1f}%")

    return risk_df, fi


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(
    X, y, df,
    cv_results, cv_oofs,
    rand_results,
    model_names,
    fi,
    risk_df,
    vol_ratio_cols,
    output_dir,
):
    print(f"\n{'='*60}")
    print("STEP 8 — Generating visualisations")
    print(f"{'='*60}")

    COLS = ["#1a56a0", "#c0392b", "#16a085", "#e67e22"]
    LBLS = [n.replace(" ", "\n") for n in model_names]
    xp   = np.arange(len(model_names))

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#f4f6fb")
    gs  = GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38,
                   left=0.05, right=0.97, top=0.93, bottom=0.05)

    def sax(ax):
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_color("#dde"); sp.set_linewidth(0.7)

    # ── 1. Correlation chart ─────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :2]); sax(ax0)
    vol_in_X = [c for c in vol_ratio_cols if c in X.columns]
    corrs = {c: stats.pointbiserialr(y, X[c])[0] for c in vol_in_X}
    corr_s = pd.Series(corrs).sort_values()
    bcolors = ["#c0392b" if v < 0 else "#1a56a0" for v in corr_s.values]
    brs = ax0.barh(range(len(corr_s)), corr_s.values, color=bcolors, height=0.72, edgecolor="white")
    ax0.set_yticks(range(len(corr_s)))
    ax0.set_yticklabels([c.replace("_", " ") for c in corr_s.index], fontsize=8)
    ax0.axvline(0, color="#333", linewidth=0.8)
    ax0.set_xlabel("Point-biserial correlation with delisting label", fontsize=9)
    ax0.set_title("Volume Feature Correlation with Delisting Label\n(ratio-engineered, redundancy removed)", fontsize=10, fontweight="bold")
    for bar, val in zip(brs, corr_s.values):
        ax0.text(val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=7.5)
    ax0.legend(handles=[
        mpatches.Patch(color="#c0392b", label="Negative (lower vol → delisting risk)"),
        mpatches.Patch(color="#1a56a0", label="Positive correlation"),
    ], fontsize=8, loc="lower right")

    # ── 2. ROC-AUC M1 vs M2 ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 2]); sax(ax1)
    m1a = [cv_results[n]["CV-AUC-mean"]     for n in model_names]
    m1s = [cv_results[n]["CV-AUC-std"]      for n in model_names]
    m2a = [rand_results[n]["ROC-AUC"]       for n in model_names]
    b1  = ax1.bar(xp - 0.2, m1a, 0.36, label="M1 CV",     color=COLS, alpha=0.9,  edgecolor="white")
    b2  = ax1.bar(xp + 0.2, m2a, 0.36, label="M2 Random", color=COLS, alpha=0.45, edgecolor="white")
    ax1.errorbar(xp - 0.2, m1a, yerr=m1s, fmt="none", color="#444", capsize=4, linewidth=1.5)
    ax1.set_xticks(xp); ax1.set_xticklabels(LBLS, fontsize=8.5)
    ax1.set_ylim(0.5, 0.82); ax1.set_ylabel("ROC-AUC", fontsize=9)
    ax1.axhline(0.5, color="#aaa", ls="--", lw=0.8)
    ax1.set_title("ROC-AUC\nMethod 1 vs Method 2", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    for bar in list(b1) + list(b2):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", fontsize=7)

    # ── 3. Precision / Recall ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 3]); sax(ax2)
    m1p = [cv_results[n]["Precision"]   for n in model_names]
    m1r = [cv_results[n]["Recall"]      for n in model_names]
    m2p = [rand_results[n]["Precision"] for n in model_names]
    m2r = [rand_results[n]["Recall"]    for n in model_names]
    w   = 0.2
    ax2.bar(xp - 1.5*w, m1p, w, label="M1 Prec",   color="#1a56a0", alpha=0.9,  edgecolor="white")
    ax2.bar(xp - 0.5*w, m1r, w, label="M1 Recall", color="#1a56a0", alpha=0.4,  edgecolor="white")
    ax2.bar(xp + 0.5*w, m2p, w, label="M2 Prec",   color="#c0392b", alpha=0.9,  edgecolor="white")
    ax2.bar(xp + 1.5*w, m2r, w, label="M2 Recall", color="#c0392b", alpha=0.4,  edgecolor="white")
    ax2.set_xticks(xp); ax2.set_xticklabels(LBLS, fontsize=8.5)
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("Score", fontsize=9)
    ax2.set_title("Precision vs Recall\n@ High-Precision Threshold", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7.5, ncol=2)

    # ── 4. PR Curves ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2]); sax(ax3)
    for i, name in enumerate(model_names):
        prec, rec, _ = precision_recall_curve(y, cv_oofs[name])
        ax3.plot(rec, prec, color=COLS[i], lw=2.2,
                 label=f"{name} (PR-AUC={cv_results[name]['PR-AUC']:.3f})")
    ax3.axhline(y.mean(), color="#aaa", ls="--", lw=1, label=f"Random baseline ({y.mean():.3f})")
    ax3.set_xlabel("Recall", fontsize=10); ax3.set_ylabel("Precision", fontsize=10)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 0.85)
    ax3.set_title("Precision–Recall Curves — All 4 Models (Method 1 CV)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)

    # ── 5. Feature importance ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:]); sax(ax4)
    fi15 = fi.head(15)

    def feat_color(name):
        if any(x in name for x in ["dev_activity", "30d_moving_avg"]): return "#9b59b6"
        if "global" in name: return "#c0392b"
        if any(x in name for x in ["dcx", "d_spot", "d_fut"]): return "#1a56a0"
        return "#16a085"

    bfc  = [feat_color(c) for c in fi15.index]
    brs2 = ax4.barh(range(len(fi15)), fi15.values, color=bfc, height=0.68, edgecolor="white")
    ax4.set_yticks(range(len(fi15)))
    ax4.set_yticklabels([c.replace("_", " ") for c in fi15.index], fontsize=8.5)
    ax4.invert_yaxis()
    ax4.set_xlabel("Feature Importance (Gain)", fontsize=9)
    ax4.set_title("Top 15 Feature Importances — XGBoost\n(volume ratios + price + Santiment dev_activity)", fontsize=10, fontweight="bold")
    for bar, val in zip(brs2, fi15.values):
        ax4.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)
    ax4.legend(handles=[
        mpatches.Patch(color="#c0392b", label="Global market signal"),
        mpatches.Patch(color="#1a56a0", label="DCX signal"),
        mpatches.Patch(color="#16a085", label="Price signal"),
        mpatches.Patch(color="#9b59b6", label="Santiment dev activity"),
    ], fontsize=8.5, loc="lower right")

    # ── 6. Metrics table ─────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :3]); ax5.axis("off")
    rows = []
    for n in model_names:
        rows.append([
            n,
            cv_results[n]["CV-AUC-mean"], cv_results[n]["PR-AUC"],
            cv_results[n]["Precision"], cv_results[n]["Recall"], cv_results[n]["F1"],
            cv_results[n]["Flagged"], cv_results[n]["TP"], cv_results[n]["FP"],
            "|",
            rand_results[n]["ROC-AUC"], rand_results[n]["PR-AUC"],
            rand_results[n]["Precision"], rand_results[n]["Recall"], rand_results[n]["F1"],
            rand_results[n]["Flagged"], rand_results[n]["TP"], rand_results[n]["FP"],
        ])
    cols = ["Model", "CV AUC", "PR-AUC", "Prec", "Rec", "F1", "#Flag", "TP", "FP", "",
            "AUC", "PR-AUC", "Prec", "Rec", "F1", "#Flag", "TP", "FP"]
    tbl = ax5.table(cellText=[[str(x) for x in r] for r in rows],
                    colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.8); tbl.scale(1.0, 2.3)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor("#1a56a0"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0: cell.set_facecolor("#eef3ff")
        cell.set_edgecolor("#dde")
        if c == 9: cell.set_facecolor("#ddd")
    ax5.text(0.29, 0.97, "Method 1 — Stratified 5-Fold CV",   transform=ax5.transAxes, ha="center", fontsize=9, fontweight="bold", color="#1a56a0")
    ax5.text(0.76, 0.97, "Method 2 — Random 80/20 Split",     transform=ax5.transAxes, ha="center", fontsize=9, fontweight="bold", color="#c0392b")
    ax5.set_title("Full Model Comparison @ High-Precision Threshold", fontsize=11, fontweight="bold", pad=30)

    # ── 7. Top 20 risk tokens ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 3]); sax(ax6)
    top20 = risk_df.head(20).sort_values("risk_score")
    bc = ["#c0392b" if s == "delisted" else "#e67e22" for s in top20["actual_status"]]
    bh = ax6.barh(range(len(top20)), top20["risk_score"], color=bc, height=0.73, edgecolor="white")
    ax6.set_yticks(range(len(top20))); ax6.set_yticklabels(top20["token"], fontsize=8.5)
    ax6.axvline(60, color="#e67e22", lw=1.5, ls="--", alpha=0.7)
    ax6.set_xlabel("Risk Score (0–100)", fontsize=9)
    ax6.set_title("Top 20 Risk Tokens\n(XGBoost)", fontsize=10, fontweight="bold")
    for bar, sc in zip(bh, top20["risk_score"]):
        ax6.text(bar.get_width() - 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{sc:.0f}", va="center", ha="right", fontsize=8.5, color="white", fontweight="bold")
    ax6.legend(handles=[
        mpatches.Patch(color="#c0392b", label="Actually delisted"),
        mpatches.Patch(color="#e67e22", label="Active (flagged)"),
    ], fontsize=8)

    fig.suptitle(
        "Token Delisting Risk Score — Volume Ratios · Correlation Selection · High-Precision Focus\nCoinDCX Hackathon 2025",
        fontsize=13, fontweight="bold", y=0.97,
    )

    out_path = os.path.join(output_dir, "model_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#f4f6fb")
    plt.close()
    print(f"  Plot saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_outputs(risk_df, cv_results, rand_results, model_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Token risk scores
    scores_path = os.path.join(output_dir, "token_risk_scores.csv")
    risk_df.to_csv(scores_path, index=False)
    print(f"  Scores saved → {scores_path}")

    # Metrics comparison
    rows = []
    for n in model_names:
        for method, src in [("Method1_CV", cv_results[n]), ("Method2_Random", rand_results[n])]:
            row = {"Model": n, "Method": method}
            row.update({k: v for k, v in src.items() if k not in ["probs", "y_te"]})
            rows.append(row)
    metrics_path = os.path.join(output_dir, "model_metrics.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"  Metrics saved → {metrics_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    df = load_data(input_path)

    # Label: 1 = delisted, 0 = active
    y = (df["status"] == "delisted").astype(int)
    print(f"\n  Class balance: {y.value_counts().to_dict()} | Imbalance ratio 1:{round(y.value_counts()[0]/y.value_counts()[1],1)}")

    # 2. Volume ratio features
    vol_ratios = engineer_volume_ratios(df)

    # 3. Correlation analysis + redundancy removal
    selected_ratios = select_volume_features(vol_ratios, y)

    # 4. Full feature matrix
    X = build_feature_matrix(df, vol_ratios, selected_ratios)
    vol_ratio_cols = selected_ratios  # for plotting

    # Models
    models = get_models()
    model_names = list(models.keys())

    # 5. Method 1 — CV
    cv_results, cv_oofs = run_method1_cv(X, y, models)

    # 6. Method 2 — Random split
    rand_results, rand_fitted, X_tr, X_te, y_tr, y_te = run_method2_random(X, y, models)

    # 7. Final scoring with XGBoost on all data
    best_model = models["XGBoost"]
    risk_df, fi = score_all_tokens(X, y, df, best_model, model_name="XGBoost")

    # 8. Plots
    plot_results(
        X, y, df,
        cv_results, cv_oofs,
        {n: {k: v for k, v in rand_results[n].items() if k not in ["probs","y_te"]}
         for n in model_names},
        model_names, fi, risk_df, vol_ratio_cols, output_dir,
    )

    # 9. Save
    print(f"\n{'='*60}")
    print("STEP 9 — Saving outputs")
    print(f"{'='*60}")
    save_outputs(risk_df, cv_results,
                 {n: {k: v for k, v in rand_results[n].items() if k not in ["probs","y_te"]}
                  for n in model_names},
                 model_names, output_dir)

    print(f"\n✓ Done. All outputs written to: {output_dir}/")
    print(f"  → token_risk_scores.csv")
    print(f"  → model_metrics.csv")
    print(f"  → model_results.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token Delisting Risk Score Model")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Path to input CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    args = parser.parse_args()
    main(args.input, args.output)
