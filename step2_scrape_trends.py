# step2_scrape_trends.py
# ─────────────────────────────────────────────────────────────────
# Pulls real Google Trends search volume for each destination and
# builds your booking-intent label (drove_intent).
#
# What it does:
#   1. Pulls 6 months of weekly search data for each destination
#   2. Detects spikes — weeks where searches are 50% above the
#      rolling 4-week average
#   3. Joins those spikes to posts — if a post was followed by a
#      spike in the next 7 days → drove_intent = 1
#
# Output:
#   data/trends.csv         weekly search volume per destination
#   data/trends_spikes.csv  each post labeled with drove_intent 0 or 1
#
# Run:
#   python step2_scrape_trends.py
# ─────────────────────────────────────────────────────────────────

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytrends.request import TrendReq
from tqdm import tqdm
from Config import (
    DESTINATIONS,
    TRENDS_LOOKBACK_MONTHS,
    TRENDS_GEO,
    POSTS_FILE,
    TRENDS_FILE,
    SPIKES_FILE,
)

os.makedirs("data", exist_ok=True)


# ── spike detection ───────────────────────────────────────────────

def detect_spike(series: pd.Series, window: int = 4, threshold: float = 1.5) -> pd.Series:
    """
    Returns a boolean Series — True means a spike was detected that week.

    How it works:
      - Compute the rolling average of the last 4 weeks
      - If this week > rolling average × 1.5 → spike

    We use a rolling average instead of the overall average to handle
    seasonality. Summer searches for Algarve are naturally higher —
    we compare against recent weeks, not the whole 6-month period.
    """
    rolling_avg = series.rolling(window=window, min_periods=1).mean()
    return series > (rolling_avg * threshold)


def spike_magnitude(series: pd.Series, window: int = 4) -> pd.Series:
    """
    How big was the spike? Measured in standard deviations above
    the rolling mean. Bigger number = stronger demand signal.

    Example: magnitude of 2.1 means the search volume was 2.1
    standard deviations above the recent average that week.
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std  = series.rolling(window=window, min_periods=1).std().fillna(1)
    return ((series - rolling_mean) / rolling_std).round(3)


# ── fetch trends ──────────────────────────────────────────────────

def fetch_trends() -> pd.DataFrame:
    """
    Pulls weekly search volume from Google Trends for every
    destination keyword in config.py.
    Returns a dataframe with one row per destination per week.
    """
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=TRENDS_LOOKBACK_MONTHS * 30)
    timeframe  = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    print(f"Fetching Google Trends")
    print(f"Timeframe : {timeframe}")
    print(f"Geo       : {TRENDS_GEO if TRENDS_GEO else 'Worldwide'}")
    print(f"Keywords  : {len(DESTINATIONS)}\n")

    pytrends = TrendReq(
        hl="en-US",
        tz=0,
        timeout=(10, 25),
        retries=3,
        backoff_factor=0.5,
    )

    all_rows = []

    for dest in tqdm(DESTINATIONS, desc="Fetching trends"):
        keyword = dest["keyword"]
        name    = dest["name"]

        try:
            pytrends.build_payload(
                kw_list=[keyword],
                timeframe=timeframe,
                geo=TRENDS_GEO,
            )

            df = pytrends.interest_over_time()

            if df.empty:
                print(f"  ⚠ No data returned for: {keyword}")
                continue

            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            series = df[keyword].astype(float)

            spikes     = detect_spike(series)
            magnitudes = spike_magnitude(series)
            rolling    = series.rolling(4, min_periods=1).mean()

            for date, value in series.items():
                all_rows.append({
                    "destination":     name,
                    "country":         dest["country"],
                    "keyword":         keyword,
                    "week":            date.strftime("%Y-%m-%d"),
                    "search_volume":   int(value),
                    "rolling_avg_4w":  round(float(rolling[date]), 2),
                    "is_spike":        bool(spikes[date]),
                    "spike_magnitude": float(magnitudes[date]),
                })

            spike_count = int(spikes.sum())
            print(f"  ✓ {name}: {len(series)} weeks | "
                  f"max={int(series.max())} | spikes={spike_count}")

            time.sleep(1.5)

        except Exception as e:
            print(f"  ✗ Error for {name}: {e}")
            time.sleep(5)

    trends_df = pd.DataFrame(all_rows)
    trends_df.to_csv(TRENDS_FILE, index=False)
    print(f"\n✅ Trends saved: {TRENDS_FILE} ({len(trends_df)} rows)")

    return trends_df


# ── join spikes to posts ──────────────────────────────────────────

def build_labels(trends_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each post in posts.csv, look at the 7-day window after it
    was posted. If any destination mentioned in the caption shows a
    spike in that window → drove_intent = 1.

    This is your label. The model learns from this.

    Example:
      Post by @visitportugal mentions "lisbon" on 2024-03-10
      Google Trends shows a spike for "flights to Lisbon" on 2024-03-14
      → drove_intent = 1
    """
    if not os.path.exists(POSTS_FILE):
        print(f"✗ {POSTS_FILE} not found — run step1 first.")
        return pd.DataFrame()

    posts_df = pd.read_csv(POSTS_FILE, parse_dates=["date"])
    trends_df["week"] = pd.to_datetime(trends_df["week"])

    print(f"\nBuilding labels for {len(posts_df)} posts...")

    results = []

    for _, post in posts_df.iterrows():
        post_date  = pd.to_datetime(post["date"])
        window_end = post_date + timedelta(days=7)

        raw_dests  = str(post.get("destinations_mentioned", ""))
        post_dests = [d.strip().lower() for d in raw_dests.split("|") if d.strip()]

        if not post_dests:
            results.append({
                "post_id":        post["post_id"],
                "creator":        post["creator"],
                "post_date":      post["date"],
                "drove_intent":   0,
                "max_spike_mag":  0.0,
                "avg_search_vol": 0.0,
                "spiked_dests":   "",
            })
            continue

        mask = (
            trends_df["destination"].str.lower().isin(post_dests) &
            (trends_df["is_spike"] == True)
        )           

        window = trends_df[mask]

        if window.empty:
            drove_intent   = 0
            max_spike_mag  = 0.0
            avg_search_vol = 0.0
            spiked_dests   = ""
        else:
            drove_intent   = int(window["is_spike"].any())
            max_spike_mag  = float(window["spike_magnitude"].max())
            avg_search_vol = float(window["search_volume"].mean())
            spiked_dests   = "|".join(
                window[window["is_spike"]]["destination"].unique()
            )

        results.append({
            "post_id":        post["post_id"],
            "creator":        post["creator"],
            "post_date":      post["date"],
            "drove_intent":   drove_intent,
            "max_spike_mag":  round(max_spike_mag, 3),
            "avg_search_vol": round(avg_search_vol, 2),
            "spiked_dests":   spiked_dests,
        })

    spikes_df  = pd.DataFrame(results)
    spikes_df.to_csv(SPIKES_FILE, index=False)

    total      = len(spikes_df)
    positives  = spikes_df["drove_intent"].sum()
    label_rate = positives / total * 100 if total > 0 else 0

    print(f"✅ Labels saved: {SPIKES_FILE} ({total} rows)")
    print(f"\nLabel summary:")
    print(f"  drove_intent = 1 : {positives} posts ({label_rate:.1f}%)")
    print(f"  drove_intent = 0 : {total - positives} posts ({100 - label_rate:.1f}%)")

    if label_rate < 5:
        print("\n  ⚠ Warning: less than 5% positive labels.")
        print("  This could mean posts are not mentioning destination keywords.")
        print("  Check DESTINATION_KEYWORDS in config.py and add more terms.")

    return spikes_df


# ── run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    trends_df = fetch_trends()

    if not trends_df.empty:
        build_labels(trends_df)

    print("\n" + "─" * 50)
    print("STEP 2 COMPLETE")
    print("─" * 50)
    print(f"  {TRENDS_FILE}")
    print(f"  {SPIKES_FILE}")