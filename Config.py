# config.py — edit this file to choose your creators and destinations

# config.py
# ─────────────────────────────────────────────────────────────────
# Control centre for the entire pipeline.
# Every other file imports settings from here.
# If you want to change anything — do it here, not inside the steps.
# ─────────────────────────────────────────────────────────────────


# ── Creators to scrape from Instagram ────────────────────────────
# Public travel accounts focused on Portugal, Ireland and Spain.
# These are AnotherTrip's core markets.

CREATORS = [
    "UC5D1TiKg5e0Q2FDzlBk2mdA",   # Visit Portugal
    "UC74Bhy3qDXmaDjM7cxwPHBg",   # Turismo de Portugal
    "UC9C_HPKwlgEnUf2Xd3oQmfA",   # Visit Spain
    "UCJe5nAWgGrII_BPfxXi8pKQ",   # Discover Ireland
    "UCV-iDbj5zWPDSn5eRh2FZfw",   # Passenger Paramvir
]


# ── Destinations to track on Google Trends ───────────────────────
# Each destination needs:
#   name        — what we call it in our data
#   keyword     — what real people type into Google when they want to book
#   country     — two-letter country code

DESTINATIONS = [
    {"name": "Lisbon",     "keyword": "flights to Lisbon",       "country": "PT"},
    {"name": "Porto",      "keyword": "visit Porto Portugal",    "country": "PT"},
    {"name": "Algarve",    "keyword": "Algarve holiday",         "country": "PT"},
    {"name": "Azores",     "keyword": "Azores travel",           "country": "PT"},
    {"name": "Dublin",     "keyword": "Dublin trip",             "country": "IE"},
    {"name": "Galway",     "keyword": "Galway Ireland visit",    "country": "IE"},
    {"name": "Seville",    "keyword": "Seville Spain travel",    "country": "ES"},
    {"name": "Barcelona",  "keyword": "Barcelona visit",         "country": "ES"},
]

# These are the keywords we look for inside captions to detect
# which destination a post is about
DESTINATION_KEYWORDS = [
    "lisbon", "porto", "portugal", "algarve", "azores",
    "dublin", "galway", "ireland", "seville", "barcelona", "spain",
    "europe", "travel",
]


# ── Scraping settings ─────────────────────────────────────────────
MAX_POSTS_PER_CREATOR  = 50   # last 50 posts per creator
MAX_COMMENTS_PER_POST  = 30   # top 30 comments per post


# ── Google Trends settings ────────────────────────────────────────
TRENDS_LOOKBACK_MONTHS = 12    # how far back to pull data
TRENDS_GEO             = ""   # "" = worldwide results
                               # "IE" = Ireland only
                               # "PT" = Portugal only


# ── Output file paths ─────────────────────────────────────────────
# All data gets saved into the data/ folder
# All trained models get saved into the models/ folder

POSTS_FILE             = "data/posts.csv"
COMMENTS_FILE          = "data/comments.csv"
TRENDS_FILE            = "data/trends.csv"
SPIKES_FILE            = "data/trends_spikes.csv"
SENTIMENT_FILE         = "data/creator_sentiment.csv"
FEATURES_FILE          = "data/features.csv"
SCORES_FILE            = "data/creator_scores.csv"
ATTRIBUTION_FILE       = "data/markov_attribution.csv"
MODEL_FILE             = "models/xgb_creator_score.json"