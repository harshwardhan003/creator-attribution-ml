# step1_scrape_youtube.py
# ─────────────────────────────────────────────────────────────────
# Scrapes real videos and comments from YouTube travel channels
# using the official YouTube Data API v3.
#
# RESUME SAFE — saves each channel immediately after scraping.
# If you stop midway and rerun, already-scraped channels are skipped.
#
# Output:
#   data/posts.csv      one row per video (same format as before)
#   data/comments.csv   one row per comment
#
# Run:
#   python step1_scrape_youtube.py
# ─────────────────────────────────────────────────────────────────

import os
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from Config import (
    CREATORS,
    MAX_POSTS_PER_CREATOR,
    MAX_COMMENTS_PER_POST,
    DESTINATION_KEYWORDS,
    POSTS_FILE,
    COMMENTS_FILE,
)

load_dotenv()
os.makedirs("data", exist_ok=True)


# ── helper functions ──────────────────────────────────────────────

def extract_destinations(text: str) -> list:
    """
    Check which destination keywords appear in a video title
    or description. Same logic as before — just different source.
    """
    if not text:
        return []
    text_lower = text.lower()
    return [kw for kw in DESTINATION_KEYWORDS if kw in text_lower]


def extract_tags(tags: list) -> list:
    """
    YouTube videos have explicit tags set by the creator.
    These are even cleaner than hashtags — creators tag intentionally.
    """
    if not tags:
        return []
    return [t.lower().strip() for t in tags]


def compute_engagement_rate(likes: int, comments: int, views: int) -> float:
    """
    For YouTube: engagement = (likes + comments) / views * 100
    Views replaces followers as the reach denominator.
    """
    if views == 0:
        return 0.0
    return round((likes + comments) / views * 100, 4)


def already_scraped(channel_id: str) -> bool:
    """
    Check if this channel was already scraped in a previous run.
    """
    if not os.path.exists(POSTS_FILE):
        return False
    try:
        existing = pd.read_csv(POSTS_FILE)
        return channel_id in existing["creator"].values
    except Exception:
        return False


def save_progress(new_posts: list, new_comments: list):
    """
    Saves after each channel finishes so progress is never lost.
    """
    if new_posts:
        posts_df = pd.DataFrame(new_posts)
        if os.path.exists(POSTS_FILE):
            existing = pd.read_csv(POSTS_FILE)
            posts_df = pd.concat([existing, posts_df], ignore_index=True)
        posts_df.to_csv(POSTS_FILE, index=False)

    if new_comments:
        comments_df = pd.DataFrame(new_comments)
        if os.path.exists(COMMENTS_FILE):
            existing = pd.read_csv(COMMENTS_FILE)
            comments_df = pd.concat([existing, comments_df], ignore_index=True)
        comments_df.to_csv(COMMENTS_FILE, index=False)


# ── YouTube API calls ─────────────────────────────────────────────

def get_channel_info(youtube, channel_id: str) -> dict:
    """
    Fetches basic channel info — name and subscriber count.
    Equivalent to Instagram's follower count and username.
    """
    response = youtube.channels().list(
        part="snippet,statistics",
        id=channel_id,
    ).execute()

    if not response.get("items"):
        return {}

    item  = response["items"][0]
    stats = item.get("statistics", {})

    return {
        "channel_name":       item["snippet"]["title"],
        "subscriber_count":   int(stats.get("subscriberCount", 0)),
        "total_video_count":  int(stats.get("videoCount", 0)),
    }


def get_channel_videos(youtube, channel_id: str, max_videos: int) -> list:
    """
    Fetches the most recent videos from a channel.
    Uses the search endpoint to get videos sorted by date.
    """
    videos    = []
    page_token = None

    while len(videos) < max_videos:
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=min(50, max_videos - len(videos)),
            order="date",          # newest first
            type="video",
            pageToken=page_token,
        )
        response = request.execute()

        for item in response.get("items", []):
            videos.append({
                "video_id":    item["id"]["videoId"],
                "title":       item["snippet"]["title"],
                "description": item["snippet"]["description"][:500],
                "published_at": item["snippet"]["publishedAt"][:10],
            })

        page_token = response.get("nextPageToken")
        if not page_token or len(videos) >= max_videos:
            break

        time.sleep(0.5)

    return videos[:max_videos]


def get_video_stats(youtube, video_ids: list) -> dict:
    """
    Fetches detailed stats for a batch of videos.
    YouTube allows up to 50 video IDs per request — very efficient.
    Returns a dict keyed by video_id.
    """
    stats = {}

    # process in batches of 50
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        response = youtube.videos().list(
            part="statistics,snippet",
            id=",".join(batch),
        ).execute()

        for item in response.get("items", []):
            s = item.get("statistics", {})
            stats[item["id"]] = {
                "views":          int(s.get("viewCount",    0)),
                "likes":          int(s.get("likeCount",    0)),
                "comments_count": int(s.get("commentCount", 0)),
                "tags":           item["snippet"].get("tags", []),
            }

        time.sleep(0.3)

    return stats


def get_video_comments(youtube, video_id: str,
                       max_comments: int) -> list:
    """
    Fetches top-level comments for a video sorted by relevance.
    Relevance sorting surfaces the most engaged comments first —
    which are more likely to contain booking intent signals.
    """
    comments   = []
    page_token = None

    try:
        while len(comments) < max_comments:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                order="relevance",
                pageToken=page_token,
            ).execute()

            for item in response.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "comment_id":    item["id"],
                    "comment_text":  top["textDisplay"][:300],
                    "comment_likes": top["likeCount"],
                    "published_at":  top["publishedAt"][:10],
                })

            page_token = response.get("nextPageToken")
            if not page_token or len(comments) >= max_comments:
                break

            time.sleep(0.3)

    except HttpError as e:
        # comments disabled on some videos — skip silently
        if "commentsDisabled" in str(e):
            pass
        else:
            print(f"    Comment error for {video_id}: {e}")

    return comments[:max_comments]


# ── main scraper ──────────────────────────────────────────────────

def scrape_youtube():
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        print("✗ YOUTUBE_API_KEY not found in .env file.")
        print("  Add: YOUTUBE_API_KEY=your_key_here")
        return

    # build the YouTube API client
    youtube = build("youtube", "v3", developerKey=api_key)
    print("YouTube API connected.\n")

    # check which channels are already done
    done = [c for c in CREATORS if already_scraped(c)]
    todo = [c for c in CREATORS if not already_scraped(c)]

    if done:
        print(f"Already scraped ({len(done)}): {len(done)} channels")
    print(f"To scrape ({len(todo)} channels)...\n")

    if not todo:
        print("All channels already scraped.")
        print("Delete data/posts.csv and data/comments.csv to rescrape.")
        return

    for channel_id in tqdm(todo, desc="Scraping channels"):
        print(f"\n→ Scraping channel {channel_id}...")

        channel_posts    = []
        channel_comments = []

        try:
            # get channel info (name, subscribers)
            info = get_channel_info(youtube, channel_id)
            if not info:
                print(f"  ✗ Channel not found — skipping")
                continue

            channel_name      = info["channel_name"]
            subscriber_count  = info["subscriber_count"]
            print(f"  Channel: {channel_name} "
                  f"({subscriber_count:,} subscribers)")

            # get recent videos
            videos = get_channel_videos(
                youtube, channel_id, MAX_POSTS_PER_CREATOR
            )
            print(f"  Found {len(videos)} videos")

            if not videos:
                continue

            # get stats for all videos in one batch request
            video_ids = [v["video_id"] for v in videos]
            stats     = get_video_stats(youtube, video_ids)

            # build one row per video
            for video in videos:
                vid_id  = video["video_id"]
                s       = stats.get(vid_id, {})

                title       = video["title"]
                description = video["description"]
                full_text   = f"{title} {description}"

                tags  = extract_tags(s.get("tags", []))
                dests = extract_destinations(full_text)
                views = s.get("views", 0)
                likes = s.get("likes", 0)
                cmnts = s.get("comments_count", 0)

                channel_posts.append({
                    # keep same column names as the old Instagram scraper
                    # so the rest of the pipeline works without changes
                    "post_id":                vid_id,
                    "creator":                channel_id,
                    "channel_name":           channel_name,
                    "date":                   video["published_at"],
                    "timestamp":              video["published_at"],
                    "likes":                  likes,
                    "comments_count":         cmnts,
                    "views":                  views,
                    "is_video":               1,
                    "caption":                full_text[:500],
                    "hashtags":               "|".join(tags),
                    "hashtag_count":          len(tags),
                    "destinations_mentioned": "|".join(dests),
                    "dest_mention_count":     len(dests),
                    "follower_count":         subscriber_count,
                    "engagement_rate":        compute_engagement_rate(
                                                likes, cmnts, views
                                              ),
                    "location_name":          None,
                    "location_lat":           None,
                    "location_lng":           None,
                })

                # get comments for this video
                comments = get_video_comments(
                    youtube, vid_id, MAX_COMMENTS_PER_POST
                )

                for c in comments:
                    channel_comments.append({
                        "post_id":       vid_id,
                        "creator":       channel_id,
                        "channel_name":  channel_name,
                        "post_date":     video["published_at"],
                        "comment_id":    c["comment_id"],
                        "comment_text":  c["comment_text"],
                        "comment_likes": c["comment_likes"],
                    })

                time.sleep(0.2)

            # save immediately after this channel finishes
            save_progress(channel_posts, channel_comments)
            print(f"  ✓ {len(channel_posts)} videos and "
                  f"{len(channel_comments)} comments saved")

            time.sleep(1)

        except HttpError as e:
            print(f"  ✗ API error for {channel_id}: {e}")
            if channel_posts:
                save_progress(channel_posts, channel_comments)

        except Exception as e:
            print(f"  ✗ Error for {channel_id}: {e}")
            if channel_posts:
                save_progress(channel_posts, channel_comments)

    # ── final summary ─────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 1 COMPLETE")
    print("─" * 50)

    if os.path.exists(POSTS_FILE):
        posts_df = pd.read_csv(POSTS_FILE)
        print(f"Videos saved:   {POSTS_FILE} ({len(posts_df)} rows)")

        if "channel_name" in posts_df.columns:
            summary = posts_df.groupby("channel_name").agg(
                videos         = ("post_id",        "count"),
                avg_views      = ("views",           "mean"),
                avg_likes      = ("likes",           "mean"),
                avg_engagement = ("engagement_rate", "mean"),
                subscribers    = ("follower_count",  "first"),
            ).round(2)
            print("\nChannel summary:")
            print(summary.to_string())

    if os.path.exists(COMMENTS_FILE):
        comments_df = pd.read_csv(COMMENTS_FILE)
        print(f"Comments saved: {COMMENTS_FILE} ({len(comments_df)} rows)")


# ── run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_youtube()