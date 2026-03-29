# run_all.py
# ─────────────────────────────────────────────────────────────────
# Master script — runs every step of the pipeline in order.
# Run this once everything is set up.
#
# What it does:
#   Step 1 → scrapes Instagram posts + comments
#   Step 2 → pulls Google Trends, builds your label
#   Step 3 → runs BERT sentiment on comments
#   Step 4 → joins everything into a feature table
#   Step 5 → trains XGBoost + Markov attribution models
#
# Usage:
#   python run_all.py                  runs all 5 steps
#   python run_all.py --skip-scraping  skips steps 1+2 if already done
#
# ─────────────────────────────────────────────────────────────────

import sys
import os
import time

SKIP_SCRAPING = "--skip-scraping" in sys.argv


def section(title: str):
    """Prints a clear section header in the terminal."""
    print("\n" + "═" * 55)
    print(f"  {title}")
    print("═" * 55)


def check_output(filepath: str) -> bool:
    """Checks if a file exists and has data in it."""
    if not os.path.exists(filepath):
        print(f"  ✗ Missing: {filepath}")
        return False
    import pandas as pd
    try:
        df = pd.read_csv(filepath)
        print(f"  ✓ {filepath} ({len(df)} rows)")
        return True
    except Exception:
        print(f"  ✗ Could not read: {filepath}")
        return False


# ─────────────────────────────────────────────────────────────────
start = time.time()
print("\n  AnotherTrip — Attribution Pipeline")
print("  Starting...\n")
# ─────────────────────────────────────────────────────────────────


# ── STEP 1: Scrape Instagram ──────────────────────────────────────
if not SKIP_SCRAPING:
    section("STEP 1 — Scraping Instagram")
    print("  Scraping posts and comments from travel creators.")
    print("  This takes ~20-30 minutes due to rate limiting.\n")
    from step1_scrape_youtube import scrape_youtube
    scrape_youtube()
else:
    section("STEP 1 — Skipped (--skip-scraping)")

check_output("data/posts.csv")
check_output("data/comments.csv")


# ── STEP 2: Google Trends ─────────────────────────────────────────
if not SKIP_SCRAPING:
    section("STEP 2 — Google Trends (building your label)")
    print("  Fetching 6 months of weekly search data.")
    print("  Detecting spikes and labeling posts.\n")
    from step2_scrape_trends import fetch_trends, build_labels
    trends_df = fetch_trends()
    if not trends_df.empty:
        build_labels(trends_df)
else:
    section("STEP 2 — Skipped (--skip-scraping)")

check_output("data/trends.csv")
check_output("data/trends_spikes.csv")


# ── STEP 3: Sentiment Analysis ────────────────────────────────────
section("STEP 3 — BERT Sentiment Analysis")
print("  Running sentiment on scraped comments.")
print("  Downloads ~500MB model on first run.\n")

from step3_sentiment import run_sentiment
run_sentiment()

check_output("data/comments_sentiment.csv")
check_output("data/creator_sentiment.csv")


# ── STEP 4: Build Features ────────────────────────────────────────
section("STEP 4 — Feature Engineering")
print("  Joining all data into one clean feature table.\n")

from step4_build_features import build_features
build_features()

check_output("data/features.csv")


# ── STEP 5: Train Models ──────────────────────────────────────────
section("STEP 5 — Training Models")
print("  Training XGBoost quality scorer.")
print("  Running Markov attribution + Shapley values.\n")

import pandas as pd
from step5_train_model import train_xgboost, run_markov

result = train_xgboost()

if result is not None:
    creator_scores, shap_importance = result
    posts_df = pd.read_csv("data/posts.csv")
    run_markov(creator_scores, posts_df)


# ── FINAL SUMMARY ─────────────────────────────────────────────────
elapsed = round(time.time() - start)
mins    = elapsed // 60
secs    = elapsed % 60

section(f"PIPELINE COMPLETE — {mins}m {secs}s")

print("\n  Output files:")
outputs = [
    "data/posts.csv",
    "data/comments.csv",
    "data/trends.csv",
    "data/trends_spikes.csv",
    "data/comments_sentiment.csv",
    "data/creator_sentiment.csv",
    "data/features.csv",
    "data/creator_scores.csv",
    "data/shap_importance.csv",
    "data/markov_attribution.csv",
    "models/xgb_model.json",
]

all_good = True
for f in outputs:
    exists = os.path.exists(f)
    status = "✓" if exists else "✗"
    if not exists:
        all_good = False
    print(f"  {status}  {f}")

if all_good:
    print("\n  All files generated successfully.")
    print("  Open the dashboard HTML to visualise your results.")
else:
    print("\n  Some files are missing — check the logs above.")