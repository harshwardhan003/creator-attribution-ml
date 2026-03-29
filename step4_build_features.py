# step4_build_features.py
# ─────────────────────────────────────────────────────────────────
# Joins all collected data into one clean feature table.
# One row per post. Every column is a number the model can learn from.
#
# Inputs:
#   data/posts.csv               from step 1
#   data/trends_spikes.csv       from step 2  (your label)
#   data/creator_sentiment.csv   from step 3
#
# Output:
#   data/features.csv            clean ML-ready feature table
#
# Run:
#   python step4_build_features.py
# ─────────────────────────────────────────────────────────────────

import os
import re
import numpy as np
import pandas as pd
from Config import (
    DESTINATION_KEYWORDS,
    POSTS_FILE,
    SPIKES_FILE,
    SENTIMENT_FILE,
    FEATURES_FILE,
)

os.makedirs("data", exist_ok=True)


# ── caption feature engineering ───────────────────────────────────

def caption_specificity(caption: str) -> float:
    """
    Scores how specific and actionable a caption is — from 0.0 to 1.0.

    A generic caption like "love this place 😍" scores low.
    A specific caption like "3 nights at this hotel in Lisbon, link in bio"
    scores high because it contains real booking signals.

    We check for:
      - Destination keyword mentions  (place names)
      - Actionable phrases            (book, hotel, tour, price)
      - Numbers                       (nights, days, prices)
      - A question mark               (invites engagement)
      - Caption length                (longer = more informative)
    """
    if not caption or len(str(caption).strip()) == 0:
        return 0.0

    caption = str(caption).lower()
    score   = 0.0

    # how many destination keywords appear
    dest_hits = sum(1 for kw in DESTINATION_KEYWORDS if kw in caption)
    score += min(dest_hits * 0.12, 0.36)

    # actionable phrases
    action_patterns = [
        r"link in bio",
        r"swipe up",
        r"book(ing)?",
        r"hotel|resort|hostel|airbnb|villa",
        r"tour|excursion|guided",
        r"\d+\s*(night|day|week)",   # "3 nights", "5 days"
        r"\€[\d,]+|\$[\d,]+",        # price mentions
        r"season|summer|winter|spring|autumn",
        r"must (visit|try|see)",
        r"hidden gem",
    ]
    for pat in action_patterns:
        if re.search(pat, caption):
            score += 0.07

    # question mark — invites comments, drives engagement
    if "?" in caption:
        score += 0.05

    # caption length — longer captions tend to be more informative
    word_count = len(caption.split())
    score += min(word_count / 150, 0.15)

    return round(min(score, 1.0), 4)


def recency_score(days_old: int) -> float:
    """
    Exponential decay — newer posts are worth more than older ones.

    Formula: e^(-days / 90)

    A post from today    → score ~1.00
    A post from 90 days  → score ~0.37
    A post from 180 days → score ~0.14
    A post from 365 days → score ~0.02

    We use 90 days as the half-life — roughly one travel booking cycle.
    """
    return round(float(np.exp(-days_old / 90)), 4)


# ── main feature builder ───────────────────────────────────────────

def build_features():

    # ── check required files exist ────────────────────────────────
    if not os.path.exists(POSTS_FILE):
        print(f"✗ {POSTS_FILE} not found — run step1 first.")
        return

    if not os.path.exists(SPIKES_FILE):
        print(f"✗ {SPIKES_FILE} not found — run step2 first.")
        return

    # ── load data ─────────────────────────────────────────────────
    posts_df = pd.read_csv(POSTS_FILE, parse_dates=["date"])
    print(f"Posts loaded:     {len(posts_df)} rows")

    spikes_df = pd.read_csv(SPIKES_FILE)
    print(f"Spike labels:     {len(spikes_df)} rows")

    # sentiment is optional — model still works without it
    if os.path.exists(SENTIMENT_FILE):
        sentiment_df = pd.read_csv(SENTIMENT_FILE)
        print(f"Creator sentiment: {len(sentiment_df)} rows")
    else:
        print("⚠ No sentiment file — run step3 for better features")
        sentiment_df = pd.DataFrame()

    # ── post-level features ───────────────────────────────────────
    print("\nEngineering features...")

    # caption specificity — how actionable is this post
    posts_df["caption_specificity"] = (
        posts_df["caption"].fillna("").apply(caption_specificity)
    )

    # days since the post was made
    today = pd.Timestamp.now().normalize()
    posts_df["days_since_post"] = (
        (today - pd.to_datetime(posts_df["date"])).dt.days
    )

    # recency score — exponential decay
    posts_df["recency_score"] = posts_df["days_since_post"].apply(recency_score)

    # log transforms — compresses skewed distributions
    # log1p means log(1 + x) so that log(0) doesn't give -infinity
    posts_df["log_likes"]     = np.log1p(posts_df["likes"].fillna(0))
    posts_df["log_comments"]  = np.log1p(posts_df["comments_count"].fillna(0))
    posts_df["log_followers"] = np.log1p(posts_df["follower_count"].fillna(0))

    # binary flag — is this a video or image
    posts_df["is_video"] = posts_df["is_video"].fillna(False).astype(int)

    # binary flag — did the creator geotag this post
    posts_df["has_geotag"] = posts_df["location_name"].notna().astype(int)

    # ── creator-level features ────────────────────────────────────
    # computed across all posts by that creator

    # posting cadence — how many posts per week on average
    def posts_per_week(dates):
        dates = pd.to_datetime(dates).sort_values()
        if len(dates) < 2:
            return 0.0
        span_days = (dates.iloc[-1] - dates.iloc[0]).days
        if span_days == 0:
            return float(len(dates))
        return round(len(dates) / (span_days / 7), 3)

    creator_stats = (
        posts_df.groupby("creator")
        .agg(
            creator_post_count       = ("post_id",         "count"),
            creator_avg_engagement   = ("engagement_rate", "mean"),
            creator_avg_specificity  = ("caption_specificity", "mean"),
            creator_geotag_rate      = ("has_geotag",      "mean"),
            creator_video_rate       = ("is_video",        "mean"),
            creator_follower_count   = ("follower_count",  "first"),
        )
        .round(4)
        .reset_index()
    )

    cadence = (
        posts_df.groupby("creator")["date"]
        .apply(posts_per_week)
        .reset_index()
    )
    cadence.columns = ["creator", "posts_per_week"]
    creator_stats = creator_stats.merge(cadence, on="creator", how="left")

    # ── join everything together ──────────────────────────────────
    features = posts_df.merge(creator_stats, on="creator", how="left")

    # join the label from step 2
    label_cols = ["post_id", "drove_intent", "max_spike_mag", "avg_search_vol"]
    available  = [c for c in label_cols if c in spikes_df.columns]
    features   = features.merge(spikes_df[available], on="post_id", how="left")
    features["drove_intent"] = features["drove_intent"].fillna(0).astype(int)

    # join sentiment features from step 3
    if not sentiment_df.empty:
        sent_cols = [
            "creator",
            "booking_intent_rate",
            "positive_rate",
            "avg_intent_score",
            "avg_sentiment_score",
        ]
        available_sent = [c for c in sent_cols if c in sentiment_df.columns]
        features = features.merge(
            sentiment_df[available_sent], on="creator", how="left"
        )
    else:
        # fill with neutral defaults if step 3 wasn't run
        features["booking_intent_rate"] = 0.0
        features["positive_rate"]       = 0.5
        features["avg_intent_score"]    = 0.0
        features["avg_sentiment_score"] = 0.5

    # fill any remaining nulls with 0
    num_cols = features.select_dtypes(include=[np.number]).columns
    features[num_cols] = features[num_cols].fillna(0)

    # ── select final columns ──────────────────────────────────────
    # keep only what the model needs — drop raw text and redundant cols
    final_cols = [
        # identifiers — not used in training, just for reference
        "post_id",
        "creator",
        "date",

        # target label — what the model is trying to predict
        "drove_intent",

        # post-level features
        "caption_specificity",
        "hashtag_count",
        "dest_mention_count",
        "has_geotag",
        "is_video",
        "recency_score",
        "days_since_post",

        # engagement (log-scaled to handle large numbers)
        "log_likes",
        "log_comments",
        "log_followers",
        "engagement_rate",

        # creator-level features
        "creator_post_count",
        "creator_avg_engagement",
        "creator_avg_specificity",
        "creator_geotag_rate",
        "creator_video_rate",
        "posts_per_week",

        # sentiment features from step 3
        "booking_intent_rate",
        "positive_rate",
        "avg_intent_score",
        "avg_sentiment_score",

        # demand signal from step 2
        "max_spike_mag",
        "avg_search_vol",
    ]

    # only keep columns that actually exist
    final_cols = [c for c in final_cols if c in features.columns]
    features   = features[final_cols].copy()

    # ── save ──────────────────────────────────────────────────────
    features.to_csv(FEATURES_FILE, index=False)

    # ── print summary ─────────────────────────────────────────────
    total     = len(features)
    positives = features["drove_intent"].sum()
    label_pct = positives / total * 100 if total > 0 else 0

    print(f"\n✅ Features saved: {FEATURES_FILE}")
    print(f"   Rows:     {total}")
    print(f"   Columns:  {len(features.columns)}")
    print(f"   Label rate (drove_intent=1): {label_pct:.1f}%")

    print("\n── All features ──────────────────────────────────────────")
    for col in features.columns:
        if col in ["post_id", "creator", "date", "drove_intent"]:
            print(f"   {col:<35} ← {'label' if col == 'drove_intent' else 'identifier'}")
        else:
            sample = features[col].dropna()
            if len(sample) > 0:
                print(f"   {col:<35} "
                      f"min={sample.min():.2f}  "
                      f"max={sample.max():.2f}  "
                      f"mean={sample.mean():.2f}")

    return features


# ── run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_features()

    print("\n" + "─" * 50)
    print("STEP 4 COMPLETE")
    print("─" * 50)
    print(f"  {FEATURES_FILE}")