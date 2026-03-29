# step3_sentiment.py
# ─────────────────────────────────────────────────────────────────
# Runs NLP sentiment analysis on scraped Instagram comments.
#
# Two layers of analysis:
#   1. Rule-based regex  — fast, detects booking intent phrases
#   2. BERT sentiment    — understands context, not just keywords
#
# Combined into a single intent_score per comment, then aggregated
# per creator so we know whose audience is most ready to book.
#
# Output:
#   data/comments_sentiment.csv   every comment with sentiment labels
#   data/creator_sentiment.csv    aggregated scores per creator
#
# Run:
#   python step3_sentiment.py
#
# Note: first run downloads the BERT model (~500MB). Cached after that.
# ─────────────────────────────────────────────────────────────────

import os
import re
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from Config import COMMENTS_FILE, SENTIMENT_FILE

os.makedirs("data", exist_ok=True)


# ── booking intent patterns ───────────────────────────────────────
# These are phrases real people write when they are considering
# booking a trip — not just admiring a photo.
# Each pattern is a regex that matches variations of the same intent.

BOOKING_INTENT_PATTERNS = [
    r"where is this",
    r"what hotel",
    r"which hotel",
    r"where (do you|did you) stay",
    r"how much.{0,20}cost",
    r"how do (i|you) get (there|to)",
    r"adding to my.{0,10}list",
    r"on my.{0,10}list",
    r"need to go",
    r"have to go",
    r"must visit",
    r"going (here|there|next)",
    r"booking (this|now|asap|it)",
    r"already booked",
    r"just booked",
    r"planning.{0,20}trip",
    r"next (holiday|vacation|trip|travel)",
    r"can you share.{0,20}(link|name|details)",
    r"what.{0,10}(tour|excursion)",
    r"recommend.{0,20}(stay|hotel|place)",
    r"how long.{0,10}stay",
    r"is it (worth|expensive|safe)",
    r"best time to (visit|go)",
]

# compile all patterns into one regex for speed
INTENT_REGEX = re.compile(
    "|".join(BOOKING_INTENT_PATTERNS),
    re.IGNORECASE
)


def has_booking_intent(text: str) -> bool:
    """
    Returns True if the comment contains a booking intent phrase.
    This is the rule-based layer — fast, no model needed.
    """
    return bool(INTENT_REGEX.search(str(text)))


def compute_intent_score(is_booking_intent: bool, sentiment: str) -> int:
    """
    Combines both signals into a single score per comment.

    Booking intent phrase found  → +2  (strong signal)
    Positive sentiment           → +1  (weak supporting signal)
    Neutral sentiment            → +0
    Negative sentiment           → -1  (negative audience reaction)

    A creator whose comments average +1.5 is far more valuable
    than one whose comments average +0.1, even if follower counts
    are similar.
    """
    score = 0
    if is_booking_intent:
        score += 2
    if sentiment == "positive":
        score += 1
    elif sentiment == "negative":
        score -= 1
    return score


# ── load BERT model ───────────────────────────────────────────────

def load_model():
    """
    Loads the cardiffnlp/twitter-roberta-base-sentiment model.

    Why this model specifically:
      - Trained on 58 million tweets
      - Designed for short informal social media text
      - Understands slang, abbreviations, emoji context
      - Perfect for Instagram comments

    First run: downloads ~500MB model and caches it locally.
    Subsequent runs: loads from cache instantly.
    """
    print("Loading BERT sentiment model...")
    print("(downloads ~500MB on first run — cached after that)\n")

    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1,       # -1 = CPU. change to 0 if you have a GPU
        truncation=True,
        max_length=128,  # comments are short — 128 tokens is plenty
        batch_size=32,   # process 32 comments at a time
    )


# ── run sentiment ─────────────────────────────────────────────────

def run_sentiment():
    if not os.path.exists(COMMENTS_FILE):
        print(f"✗ {COMMENTS_FILE} not found — run step1 first.")
        return

    comments_df = pd.read_csv(COMMENTS_FILE)
    total = len(comments_df)

    print(f"Loaded {total} comments from "
          f"{comments_df['creator'].nunique()} creators\n")

    # ── layer 1: rule-based booking intent ───────────────────────
    # fast — runs on every comment in seconds
    print("Layer 1: detecting booking intent phrases...")
    comments_df["is_booking_intent"] = (
        comments_df["comment_text"]
        .fillna("")
        .apply(has_booking_intent)
    )
    intent_count = comments_df["is_booking_intent"].sum()
    print(f"  Found {intent_count} booking intent comments "
          f"({intent_count/total*100:.1f}%)\n")

    # ── layer 2: BERT sentiment ───────────────────────────────────
    # slower — processes in batches of 32
    classifier = load_model()
    texts      = comments_df["comment_text"].fillna("").tolist()

    print(f"Layer 2: running BERT sentiment on {total} comments...")
    sentiment_labels = []
    sentiment_scores = []
    batch_size = 32

    for i in tqdm(range(0, total, batch_size), desc="Sentiment batches"):
        batch = texts[i : i + batch_size]
        try:
            results = classifier(batch)
            for r in results:
                sentiment_labels.append(r["label"].lower())
                sentiment_scores.append(round(r["score"], 4))
        except Exception as e:
            # if a batch fails fill with neutral so we don't lose rows
            print(f"\n  ⚠ Batch failed: {e} — filling with neutral")
            for _ in batch:
                sentiment_labels.append("neutral")
                sentiment_scores.append(0.5)

    comments_df["sentiment_label"] = sentiment_labels
    comments_df["sentiment_score"]  = sentiment_scores

    # ── combine into intent_score ─────────────────────────────────
    comments_df["intent_score"] = comments_df.apply(
        lambda row: compute_intent_score(
            row["is_booking_intent"],
            row["sentiment_label"]
        ),
        axis=1
    )

    # save the full comment-level results
    comments_df.to_csv("data/comments_sentiment.csv", index=False)
    print(f"\n✅ Saved: data/comments_sentiment.csv ({total} rows)")

    # ── aggregate per creator ─────────────────────────────────────
    # this is what we actually use as features in the model
    creator_agg = (
        comments_df
        .groupby("creator")
        .agg(
            total_comments       = ("comment_text",      "count"),
            booking_intent_count = ("is_booking_intent", "sum"),
            booking_intent_rate  = ("is_booking_intent", "mean"),
            positive_rate        = ("sentiment_label",   lambda x: (x == "positive").mean()),
            negative_rate        = ("sentiment_label",   lambda x: (x == "negative").mean()),
            avg_sentiment_score  = ("sentiment_score",   "mean"),
            avg_intent_score     = ("intent_score",      "mean"),
            total_intent_score   = ("intent_score",      "sum"),
        )
        .round(4)
        .reset_index()
        .sort_values("avg_intent_score", ascending=False)
    )

    creator_agg.to_csv("data/creator_sentiment.csv", index=False)
    print(f"✅ Saved: data/creator_sentiment.csv ({len(creator_agg)} rows)")

    # ── print summary ─────────────────────────────────────────────
    print("\n── Creator sentiment summary ────────────────────────────")
    print(creator_agg[[
        "creator",
        "total_comments",
        "booking_intent_rate",
        "positive_rate",
        "avg_intent_score",
    ]].to_string(index=False))

    return comments_df, creator_agg


# ── run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_sentiment()

    print("\n" + "─" * 50)
    print("STEP 3 COMPLETE")
    print("─" * 50)
    print("  data/comments_sentiment.csv")
    print("  data/creator_sentiment.csv")