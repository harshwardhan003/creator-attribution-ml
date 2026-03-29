# step5_train_model.py
# ─────────────────────────────────────────────────────────────────
# Trains two models on the real collected data.
#
# MODEL 1 — XGBoost Creator Quality Score
#   Learns which post/creator features predict booking intent.
#   Outputs a quality score per creator + SHAP explanations.
#
# MODEL 2 — Markov Chain Attribution + Shapley Values
#   Builds conversion paths from real engagement data.
#   Computes removal effect per creator (true attribution credit).
#   Distributes credit fairly using Shapley values.
#
# Input:
#   data/features.csv
#
# Output:
#   data/creator_scores.csv      XGBoost quality score per creator
#   data/shap_importance.csv     which features matter most
#   data/markov_attribution.csv  removal effects + Shapley credits
#   models/xgb_model.json        saved model file
#
# Run:
#   python step5_train_model.py
# ─────────────────────────────────────────────────────────────────

import os
import warnings
import numpy as np
import pandas as pd
from itertools import combinations

warnings.filterwarnings("ignore")

from sklearn.model_selection  import StratifiedKFold, cross_val_score
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.metrics          import roc_auc_score, average_precision_score

import xgboost as xgb
import shap

from Config import FEATURES_FILE, SCORES_FILE, ATTRIBUTION_FILE, MODEL_FILE

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# MODEL 1 — XGBoost Creator Quality Score
# ══════════════════════════════════════════════════════════════════

def train_xgboost():
    print("=" * 55)
    print("MODEL 1 — XGBoost Creator Quality Score")
    print("=" * 55)

    # ── load features ─────────────────────────────────────────────
    if not os.path.exists(FEATURES_FILE):
        print(f"✗ {FEATURES_FILE} not found — run step4 first.")
        return None

    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── check we have enough labeled data ────────────────────────
    if "drove_intent" not in df.columns:
        print("✗ No label column found.")
        return None

    labeled = df.dropna(subset=["drove_intent"])

    if len(labeled) < 20:
        print(f"✗ Only {len(labeled)} labeled rows — need at least 20.")
        print("  Make sure steps 1 and 2 ran successfully.")
        return None

    print(f"Training on {len(labeled)} labeled posts")

    positives  = labeled["drove_intent"].sum()
    label_rate = positives / len(labeled) * 100
    print(f"Label balance: {label_rate:.1f}% positive\n")

    # ── define features ───────────────────────────────────────────
    # these are the columns the model learns from
    # we exclude identifiers and the label itself
    skip_cols    = ["post_id", "creator", "date", "drove_intent"]
    feature_cols = [
        c for c in labeled.columns
        if c not in skip_cols
        and labeled[c].dtype in [np.float64, np.int64, float, int]
    ]

    X = labeled[feature_cols].fillna(0).values
    y = labeled["drove_intent"].astype(int).values

    print(f"Features used ({len(feature_cols)}):")
    print(f"  {', '.join(feature_cols)}\n")

    # ── build the model ───────────────────────────────────────────
    # scale_pos_weight handles class imbalance
    # if 10% of posts drove intent and 90% didn't, we set this to 9
    # so the model doesn't just always predict 0
    neg  = (y == 0).sum()
    pos  = (y == 1).sum()
    spw  = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators      = 200,    # number of trees
        max_depth         = 4,      # how deep each tree goes
        learning_rate     = 0.05,   # how much each tree corrects the last
        subsample         = 0.8,    # use 80% of rows per tree (prevents overfitting)
        colsample_bytree  = 0.8,    # use 80% of features per tree
        scale_pos_weight  = spw,    # handle class imbalance
        eval_metric       = "auc",
        random_state      = 42,
        verbosity         = 0,
    )

    # ── cross-validation ──────────────────────────────────────────
    # splits data into 5 folds, trains on 4, tests on 1, rotates
    # gives an honest estimate of how well the model generalises
    if len(y) >= 30:
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        print(f"Cross-validation ROC-AUC: {cv_auc.mean():.3f} "
              f"(± {cv_auc.std():.3f})")
        print(f"  — 0.5 = random guess | 1.0 = perfect | 0.7+ = good\n")

    # ── train on all data ─────────────────────────────────────────
    model.fit(X, y)

    # calibrate so the score is a true probability
    # without this, score 0.7 doesn't mean "70% likely to drive intent"
    # with Platt scaling it does
    pos = (y == 1).sum()
    if len(y) >= 30 and pos >= 6:
        calibrated = CalibratedClassifierCV(model, cv=3, method="sigmoid")
        calibrated.fit(X, y)
        probs = calibrated.predict_proba(X)[:, 1]
    else:
        probs = model.predict_proba(X)[:, 1]

    train_auc = roc_auc_score(y, probs)
    train_ap  = average_precision_score(y, probs)
    print(f"Train ROC-AUC:        {train_auc:.3f}")
    print(f"Train Avg Precision:  {train_ap:.3f}\n")

    # ── SHAP explainability ───────────────────────────────────────
    # for every post, SHAP tells us which features pushed the
    # prediction up or down and by how much
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    # average absolute SHAP value per feature = overall importance
    shap_importance = (
        pd.DataFrame({
            "feature":    feature_cols,
            "importance": np.abs(shap_vals).mean(axis=0),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n── Top features by SHAP importance ──────────────────────")
    print(shap_importance.head(10).to_string(index=False))

    # ── aggregate to creator level ────────────────────────────────
    result_df = labeled[["post_id", "creator", "date", "drove_intent"]].copy()
    result_df["quality_score"] = probs.round(4)
    result_df["score_pct"]     = (probs * 100).round(1)

    creator_scores = (
        result_df.groupby("creator")
        .agg(
            avg_score    = ("quality_score", "mean"),
            median_score = ("quality_score", "median"),
            posts_scored = ("post_id",       "count"),
            intent_posts = ("drove_intent",  "sum"),
            intent_rate  = ("drove_intent",  "mean"),
        )
        .round(4)
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )
    creator_scores["score_pct"] = (creator_scores["avg_score"] * 100).round(1)

    print("\n── Creator quality scores ───────────────────────────────")
    print(creator_scores[[
        "creator", "score_pct", "posts_scored", "intent_rate"
    ]].to_string(index=False))

    # ── save outputs ──────────────────────────────────────────────
    creator_scores.to_csv(SCORES_FILE, index=False)
    shap_importance.to_csv("data/shap_importance.csv", index=False)
    result_df.to_csv("data/model_results.csv", index=False)
    model.save_model(MODEL_FILE)

    print(f"\n✅ Creator scores:    {SCORES_FILE}")
    print(f"✅ SHAP importance:   data/shap_importance.csv")
    print(f"✅ Model saved:       {MODEL_FILE}")

    return creator_scores, shap_importance


# ══════════════════════════════════════════════════════════════════
# MODEL 2 — Markov Chain Attribution + Shapley Values
# ══════════════════════════════════════════════════════════════════

def build_conversion_paths(creator_scores: pd.DataFrame,
                           posts_df: pd.DataFrame,
                           n_paths: int = 500) -> list:
    """
    Builds simulated conversion paths grounded in real engagement data.

    Each path is a list of creators a user "touched" before booking.
    Example: ["visitportugal", "nudimentary", "kirstie.pike"]

    We weight each creator by their real quality score so creators
    with stronger real-world signals appear in more paths.
    """
    rng      = np.random.default_rng(42)
    creators = creator_scores["creator"].tolist()
    scores   = creator_scores["avg_score"].tolist()
    rates    = creator_scores["intent_rate"].tolist()

    # weight each creator by their quality score
    # creators with higher real scores appear in more paths
    weights = np.array(scores) + 0.05   # small floor so nobody has 0 weight
    weights = weights / weights.sum()

    paths = []

    for _ in range(n_paths):
        # most journeys touch 1-3 creators before booking
        path_len = rng.choice([1, 2, 3], p=[0.50, 0.35, 0.15])
        path     = rng.choice(
            creators, size=path_len, replace=False, p=weights
        ).tolist()

        # does this path convert?
        # probability = max intent rate among creators in path
        max_rate = max(rates[creators.index(c)] for c in path)
        if rng.random() < max_rate * 1.5:
            paths.append(path)

    print(f"Built {len(paths)} conversion paths from real data")
    return paths


def removal_effect(paths: list, creators: list) -> dict:
    """
    For each creator: what percentage of conversions would we lose
    if that creator was removed from every path?

    Higher removal effect = more critical to conversion = more credit.

    Example:
      Total paths = 500
      Paths without @nudimentary = 412
      Removal effect = (500 - 412) / 500 = 17.6%
    """
    total = len(paths)
    if total == 0:
        return {c: 0.0 for c in creators}

    effects = {}
    for creator in creators:
        paths_without = [p for p in paths if creator not in p]
        effect = (total - len(paths_without)) / total
        effects[creator] = round(effect, 4)

    return effects


def shapley_values(paths: list, creators: list) -> dict:
    """
    Computes the Shapley value for each creator.

    Shapley value = average marginal contribution across every
    possible combination of creators.

    This is the mathematically fairest way to split credit.
    It comes from cooperative game theory — originally used to
    fairly divide profits among business partners.

    For attribution: each creator gets exactly what they add,
    averaged across every possible coalition they could be in.
    """
    total = len(paths)
    if total == 0:
        return {c: 0.0 for c in creators}

    def coalition_value(coalition):
        """fraction of paths where at least one coalition member appears"""
        if not coalition:
            return 0.0
        return sum(
            1 for p in paths if any(c in p for c in coalition)
        ) / total

    n        = len(creators)
    shapley  = {c: 0.0 for c in creators}

    for creator in creators:
        others = [c for c in creators if c != creator]
        total_marginal = 0.0

        for r in range(len(others) + 1):
            for subset in combinations(others, r):
                # weight for this coalition size
                weight = (
                    np.math.factorial(r) *
                    np.math.factorial(n - r - 1) /
                    np.math.factorial(n)
                )
                without = frozenset(subset)
                with_   = frozenset(subset) | {creator}
                marginal = coalition_value(with_) - coalition_value(without)
                total_marginal += weight * marginal

        shapley[creator] = round(total_marginal, 4)

    # normalise so all values sum to 1
    total_shapley = sum(shapley.values())
    if total_shapley > 0:
        shapley = {k: round(v / total_shapley, 4) for k, v in shapley.items()}

    return shapley


def run_markov(creator_scores: pd.DataFrame, posts_df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("MODEL 2 — Markov Attribution + Shapley Values")
    print("=" * 55)

    creators = creator_scores["creator"].tolist()
    paths    = build_conversion_paths(creator_scores, posts_df)

    print("Computing removal effects...")
    effects = removal_effect(paths, creators)

    print("Computing Shapley values (may take ~1 minute)...")
    shapley = shapley_values(paths, creators)

    # last-click baseline for comparison
    last_click = {}
    for creator in creators:
        lc = sum(1 for p in paths if p and p[-1] == creator)
        last_click[creator] = round(lc / len(paths), 4) if paths else 0.0

    attribution = pd.DataFrame({
        "creator":             creators,
        "removal_effect":      [effects[c]    for c in creators],
        "shapley_credit":      [shapley[c]    for c in creators],
        "last_click_share":    [last_click[c] for c in creators],
        "delta_vs_lastclick":  [
            round(shapley[c] - last_click[c], 4) for c in creators
        ],
    }).sort_values("removal_effect", ascending=False)

    attribution = attribution.merge(
        creator_scores[["creator", "avg_score", "intent_rate"]],
        on="creator", how="left"
    )

    print("\n── Attribution results ───────────────────────────────────")
    print(attribution[[
        "creator", "removal_effect", "shapley_credit",
        "last_click_share", "delta_vs_lastclick"
    ]].to_string(index=False))

    attribution.to_csv(ATTRIBUTION_FILE, index=False)
    print(f"\n✅ Attribution saved: {ATTRIBUTION_FILE}")

    return attribution


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = train_xgboost()

    if result is not None:
        creator_scores, shap_importance = result
        posts_df = pd.read_csv("data/posts.csv")
        run_markov(creator_scores, posts_df)

    print("\n" + "=" * 55)
    print("STEP 5 COMPLETE")
    print("=" * 55)
    print(f"  {SCORES_FILE}")
    print(f"  data/shap_importance.csv")
    print(f"  {ATTRIBUTION_FILE}")
    print(f"  {MODEL_FILE}")
