# app.py
# ─────────────────────────────────────────────────────────────────
# AnotherTrip Attribution Intelligence Dashboard
# Run: streamlit run app.py
# ─────────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AnotherTrip Attribution",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── design system ─────────────────────────────────────────────────
# one palette, used everywhere — no random greens/oranges
PRIMARY    = "#0a0f1e"   # deep navy — headings, dark bars
ACCENT     = "#2563eb"   # electric blue — highlights
ACCENT2    = "#60a5fa"   # lighter blue — secondary
BG         = "#f8fafc"   # off-white page background
CARD_BG    = "#ffffff"   # white cards
BORDER     = "#e2e8f0"   # subtle borders
TEXT       = "#0f172a"   # primary text
MUTED      = "#64748b"   # secondary text
SCALE      = ["#dbeafe", "#93c5fd", "#3b82f6", "#1d4ed8", "#0a0f1e"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Instrument+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Instrument Sans', sans-serif;
    background-color: {BG};
}}

h1, h2, h3 {{
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: {TEXT} !important;
}}

/* sidebar */
[data-testid="stSidebar"] {{
    background: {PRIMARY};
    border-right: none;
}}
[data-testid="stSidebar"] * {{ color: #cbd5e1 !important; }}
[data-testid="stSidebar"] .stRadio label {{
    font-size: 13.5px;
    padding: 5px 0;
    transition: color 0.15s;
}}

/* metric cards */
[data-testid="metric-container"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 20px !important;
    border-top: 3px solid {ACCENT};
}}
[data-testid="metric-container"] label {{
    color: {MUTED} !important;
    font-size: 11px !important;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}}

/* cards */
.at-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 14px;
    height: 100%;
}}

.at-card-top {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-top: 3px solid {ACCENT};
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 14px;
    text-align: center;
}}

.hero-card {{
    background: {PRIMARY};
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
}}

.section-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 12px;
    margin-top: 4px;
}}

.page-title {{
    font-family: 'Syne', sans-serif;
    font-size: 30px;
    font-weight: 800;
    color: {TEXT};
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}}

.page-sub {{
    font-size: 15px;
    color: {MUTED};
    margin-bottom: 28px;
    line-height: 1.6;
}}

.stat-number {{
    font-family: 'Syne', sans-serif;
    font-size: 34px;
    font-weight: 800;
    color: {ACCENT};
    line-height: 1;
    margin: 8px 0 4px;
}}

.stat-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: {MUTED};
}}

.stat-sub {{
    font-size: 12px;
    color: {MUTED};
    margin-top: 4px;
}}

.pill {{
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}}

.shap-bar {{
    height: 6px;
    background: {ACCENT};
    border-radius: 3px;
    margin-top: 5px;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
}}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    files = {
        "posts":       "data/posts.csv",
        "attribution": "data/markov_attribution.csv",
        "scores":      "data/creator_scores.csv",
        "shap":        "data/shap_importance.csv",
        "sentiment":   "data/creator_sentiment.csv",
        "features":    "data/features.csv",
        "trends":      "data/trends.csv",
    }
    return {k: pd.read_csv(v) if os.path.exists(v) else pd.DataFrame()
            for k, v in files.items()}


def name(channel_id, posts_df):
    if posts_df.empty or "channel_name" not in posts_df.columns:
        return channel_id[:14] + "..."
    m = posts_df[posts_df["creator"] == channel_id]["channel_name"]
    return m.iloc[0] if not m.empty else channel_id[:14] + "..."


def chart_layout(fig, height=280):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=8, b=0),
        height=height,
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        font=dict(family="Instrument Sans", color=MUTED, size=12),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9",
                     zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=False, zeroline=False,
                     tickfont=dict(color=MUTED))
    return fig


# ── sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding: 24px 0 32px 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 22px;
                    font-weight: 800; color: white; letter-spacing: -0.5px;'>
            another<span style='color: #60a5fa;'>trip</span>
        </div>
        <div style='font-size: 10px; color: #475569; letter-spacing: 2px;
                    text-transform: uppercase; margin-top: 5px;'>
            Attribution Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "Overview",
        "Creator Quality Scores",
        "SHAP Explainability",
        "Markov Attribution",
        "Sentiment Analysis",
        "Payout Calculator",
    ])

    st.markdown(f"""
    <div style='margin-top: 32px; padding-top: 20px;
                border-top: 1px solid #1e293b;'>
        <div style='font-size: 10px; color: #334155; font-weight: 600;
                    letter-spacing: 1px; text-transform: uppercase;
                    margin-bottom: 10px;'>Data sources</div>
        <div style='font-size: 12px; color: #475569; line-height: 2;'>
            YouTube Data API v3<br>
            Google Trends<br>
            BERT (HuggingFace)
        </div>
        <div style='font-size: 10px; color: #334155; font-weight: 600;
                    letter-spacing: 1px; text-transform: uppercase;
                    margin: 16px 0 10px;'>Models</div>
        <div style='font-size: 12px; color: #475569; line-height: 2;'>
            XGBoost + SHAP<br>
            Markov Chain<br>
            Shapley Values
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── load ──────────────────────────────────────────────────────────
data       = load_data()
posts      = data["posts"]
attribution = data["attribution"]
scores     = data["scores"]
shap       = data["shap"]
sentiment  = data["sentiment"]
features   = data["features"]
trends     = data["trends"]

for df_name in ["attribution", "scores", "sentiment"]:
    df = data[df_name]
    if not df.empty and not posts.empty:
        data[df_name]["name"] = data[df_name]["creator"].apply(
            lambda x: name(x, posts))

attribution = data["attribution"]
scores      = data["scores"]
sentiment   = data["sentiment"]


# ══════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="page-title">Attribution Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Which travel creators actually drive bookings — not just who gets the last click.</div>', unsafe_allow_html=True)

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    comments_count = len(pd.read_csv("data/comments.csv")) if os.path.exists("data/comments.csv") else 0
    with c1: st.metric("Videos scraped",   f"{len(posts):,}")
    with c2: st.metric("Comments analysed", f"{comments_count:,}")
    with c3: st.metric("Creators tracked",  posts["creator"].nunique() if not posts.empty else 0)
    with c4: st.metric("Avg engagement",    f"{posts['engagement_rate'].mean():.2f}%" if not posts.empty else "—")
    with c5: st.metric("Top creator",       attribution.iloc[0]["name"] if not attribution.empty else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # hero insight
    if not attribution.empty:
        top = attribution.iloc[0]
        st.markdown(f"""
        <div class="hero-card">
            <div style='font-size: 10px; color: #60a5fa; letter-spacing: 2px;
                        text-transform: uppercase; margin-bottom: 12px;'>
                ⚡ KEY INSIGHT
            </div>
            <div style='font-family: Syne, sans-serif; font-size: 22px;
                        font-weight: 700; color: white; margin-bottom: 10px;
                        line-height: 1.3;'>
                {top['name']} is responsible for {top['removal_effect']*100:.1f}%
                of all attributed conversions
            </div>
            <div style='font-size: 14px; color: #94a3b8; line-height: 1.7;'>
                Under last-click attribution they receive {top['last_click_share']*100:.1f}%
                of credit. Shapley values — which fairly distribute credit across all
                creators in a conversion path — give them {top['shapley_credit']*100:.1f}%.
            </div>
        </div>
        """, unsafe_allow_html=True)

    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="section-label">Engagement rate by creator</div>', unsafe_allow_html=True)
        if not posts.empty and "channel_name" in posts.columns:
            eng = posts.groupby("channel_name")["engagement_rate"].mean().reset_index()
            eng.columns = ["Creator", "Engagement (%)"]
            eng = eng.sort_values("Engagement (%)", ascending=True)
            fig = px.bar(eng, x="Engagement (%)", y="Creator",
                         orientation="h",
                         color="Engagement (%)",
                         color_continuous_scale=SCALE)
            chart_layout(fig, 260)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="section-label">Videos published over time</div>', unsafe_allow_html=True)
        if not posts.empty:
            posts["date"] = pd.to_datetime(posts["date"])
            m = posts.groupby(posts["date"].dt.to_period("M")).size().reset_index()
            m.columns = ["Month","Videos"]
            m["Month"] = m["Month"].astype(str)
            fig2 = px.area(m, x="Month", y="Videos",
                           color_discrete_sequence=[ACCENT])
            chart_layout(fig2, 260)
            fig2.update_traces(line_width=2.5,
                               fillcolor=f"rgba(37,99,235,0.08)")
            st.plotly_chart(fig2, use_container_width=True)

    # pipeline steps
    st.markdown("---")
    st.markdown('<div class="section-label">Pipeline overview</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Collect",   "YouTube videos + comments from 5 travel creators"),
        ("02", "Label",     "Google Trends spike detection — real demand signal"),
        ("03", "Analyse",   "BERT sentiment on comments for booking intent"),
        ("04", "Model",     "XGBoost on 23 features predicts booking probability"),
        ("05", "Attribute", "Markov + Shapley fairly split credit across creators"),
    ]
    cols = st.columns(5)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="at-card" style='text-align: center; border-top: 3px solid {ACCENT};'>
                <div style='font-family: Syne, sans-serif; font-size: 26px;
                            font-weight: 800; color: #e2e8f0;'>{num}</div>
                <div style='font-weight: 600; font-size: 13px;
                            color: {TEXT}; margin: 8px 0 5px;'>{title}</div>
                <div style='font-size: 12px; color: {MUTED};
                            line-height: 1.55;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CREATOR QUALITY SCORES
# ══════════════════════════════════════════════════════════════════
elif page == "Creator Quality Scores":
    st.markdown('<div class="page-title">Creator Quality Scores</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">XGBoost model trained on 23 features — predicts which creators drive real booking demand, not just views.</div>', unsafe_allow_html=True)

    if scores.empty:
        st.warning("Run step5_train_model.py first.")
    else:
        cols = st.columns(len(scores))
        for i, (_, row) in enumerate(scores.iterrows()):
            with cols[i]:
                sc = row["score_pct"]
                st.markdown(f"""
                <div class="at-card-top">
                    <div class="stat-label">{row["name"]}</div>
                    <div class="stat-number">{sc:.1f}%</div>
                    <div class="stat-sub">{int(row['posts_scored'])} videos</div>
                    <div class="stat-sub">{row['intent_rate']*100:.1f}% intent rate</div>
                    <div style='margin-top: 12px;'>
                        <div style='background: #f1f5f9; border-radius: 4px;
                                    height: 5px; overflow: hidden;'>
                            <div style='height: 5px; border-radius: 4px;
                                        width: {min(sc*8, 100):.0f}%;
                                        background: {ACCENT};'></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cl, cr = st.columns(2)

        with cl:
            st.markdown('<div class="section-label">Score comparison</div>', unsafe_allow_html=True)
            fig = px.bar(scores.sort_values("score_pct", ascending=True),
                         x="score_pct", y="name", orientation="h",
                         color="score_pct", color_continuous_scale=SCALE,
                         labels={"score_pct": "Quality Score (%)", "name": ""})
            chart_layout(fig, 280)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.markdown('<div class="section-label">Score vs engagement rate</div>', unsafe_allow_html=True)
            if not posts.empty:
                mg = scores.merge(
                    posts.groupby("creator")["engagement_rate"].mean().reset_index(),
                    on="creator", how="left")
                fig2 = px.scatter(mg, x="engagement_rate", y="score_pct",
                                  text="name", size="score_pct", size_max=32,
                                  color="score_pct", color_continuous_scale=SCALE,
                                  labels={"engagement_rate": "Engagement Rate (%)",
                                          "score_pct": "Quality Score (%)"})
                fig2.update_traces(textposition="top center", textfont_size=11,
                                   textfont_color=MUTED, marker_line_width=0)
                chart_layout(fig2, 280)
                fig2.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown(f"""
        <div class="at-card">
            <div class="section-label" style='margin-bottom: 8px;'>What this score means</div>
            <p style='font-size: 14px; color: {MUTED}; line-height: 1.8; margin: 0;'>
                The quality score is the model's predicted probability that a creator's
                content drives a measurable spike in destination search demand within 7 days.
                It is calibrated using Platt scaling — so a score of <b style='color:{TEXT}'>8.5%</b>
                means historically creators at this score drove demand spikes on 8.5% of their posts.
                The model was trained on 23 features including caption specificity,
                BERT comment intent rate, engagement rate, and Google Trends spike magnitude.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════
elif page == "SHAP Explainability":
    st.markdown('<div class="page-title">SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Why did the model score each creator the way it did? SHAP values reveal which features matter most.</div>', unsafe_allow_html=True)

    if shap.empty:
        st.warning("Run step5_train_model.py first.")
    else:
        cl, cr = st.columns([3, 2])

        with cl:
            st.markdown('<div class="section-label">Feature importance — mean |SHAP value|</div>', unsafe_allow_html=True)
            s = shap.sort_values("importance", ascending=True).tail(12)
            fig = px.bar(s, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale=SCALE,
                         labels={"importance": "Mean |SHAP value|", "feature": ""})
            chart_layout(fig, 400)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.markdown('<div class="section-label">Feature explanations</div>', unsafe_allow_html=True)
            explanations = {
                "max_spike_mag":       "Size of the Google Trends demand spike after the post",
                "avg_search_vol":      "Average search volume for the destination",
                "booking_intent_rate": "% of comments showing booking intent (BERT)",
                "log_likes":           "Log-scaled likes — handles large numbers fairly",
                "recency_score":       "Exponential decay — newer posts score higher",
                "hashtag_count":       "Number of hashtags in the video title/description",
                "log_comments":        "Log-scaled comment count",
                "days_since_post":     "How many days ago the video was published",
                "dest_mention_count":  "How many destination keywords appear in content",
                "avg_sentiment_score": "Average BERT confidence score across comments",
            }
            for feat, desc in explanations.items():
                match = shap[shap["feature"] == feat]
                if not match.empty:
                    imp = match.iloc[0]["importance"]
                    bar_w = min(imp / shap["importance"].max() * 100, 100)
                    st.markdown(f"""
                    <div style='padding: 10px 14px; border: 1px solid {BORDER};
                                border-radius: 8px; margin-bottom: 8px;
                                background: {CARD_BG};'>
                        <div style='display: flex; justify-content: space-between;
                                    align-items: center; margin-bottom: 3px;'>
                            <span style='font-size: 12px; font-weight: 600;
                                         color: {TEXT};'>{feat}</span>
                            <span style='font-size: 11px; color: {ACCENT};
                                         font-weight: 600;'>{imp:.3f}</span>
                        </div>
                        <div style='background: #f1f5f9; border-radius: 3px;
                                    height: 3px; margin-bottom: 5px;'>
                            <div style='height: 3px; border-radius: 3px;
                                        width: {bar_w:.0f}%;
                                        background: {ACCENT};'></div>
                        </div>
                        <div style='font-size: 11px; color: {MUTED};'>{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

        top3 = shap.nlargest(3, "importance")["feature"].tolist()
        st.markdown(f"""
        <div class="at-card" style='border-top: 3px solid {ACCENT};'>
            <div class="section-label">Plain English takeaway</div>
            <p style='font-size: 14px; color: {MUTED}; line-height: 1.8; margin: 0;'>
                The three most important signals are <b style='color:{TEXT}'>{top3[0]}</b>,
                <b style='color:{TEXT}'>{top3[1]}</b>, and <b style='color:{TEXT}'>{top3[2]}</b>.
                The model is primarily driven by actual demand signals — Google Trends spikes —
                rather than vanity metrics like follower count or raw view counts.
                A creator with modest subscribers posting highly specific destination content
                will score higher than a large creator posting generic travel content.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MARKOV ATTRIBUTION
# ══════════════════════════════════════════════════════════════════
elif page == "Markov Attribution":
    st.markdown('<div class="page-title">Markov Attribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Fairly distributing booking credit across creators using Markov chains and Shapley values from cooperative game theory.</div>', unsafe_allow_html=True)

    if attribution.empty:
        st.warning("Run step5_train_model.py first.")
    else:
        top = attribution.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Top removal effect",    f"{top['removal_effect']*100:.1f}%")
        with c2: st.metric("Top Shapley credit",    f"{top['shapley_credit']*100:.1f}%")
        with c3:
            md = attribution["delta_vs_lastclick"].abs().max()
            st.metric("Max last-click error", f"{md*100:.1f}pp")
        with c4: st.metric("Creators modelled", len(attribution))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Shapley credit vs last-click — side by side</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Last-click (naive)",
            x=attribution["name"],
            y=attribution["last_click_share"] * 100,
            marker_color="#e2e8f0", marker_line_width=0,
        ))
        fig.add_trace(go.Bar(
            name="Shapley credit (fair)",
            x=attribution["name"],
            y=attribution["shapley_credit"] * 100,
            marker_color=ACCENT, marker_line_width=0,
        ))
        chart_layout(fig, 300)
        fig.update_layout(
            barmode="group",
            yaxis_title="Credit (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        cl, cr = st.columns(2)

        with cl:
            st.markdown('<div class="section-label">Removal effect per creator</div>', unsafe_allow_html=True)
            fig2 = px.bar(attribution.sort_values("removal_effect", ascending=True),
                          x="removal_effect", y="name", orientation="h",
                          color="removal_effect", color_continuous_scale=SCALE,
                          labels={"removal_effect": "Removal Effect", "name": ""})
            chart_layout(fig2, 240)
            fig2.update_layout(coloraxis_showscale=False)
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(fig2, use_container_width=True)

        with cr:
            st.markdown('<div class="section-label">Over/under credited vs last-click</div>', unsafe_allow_html=True)
            colors = [ACCENT if v > 0 else "#cbd5e1"
                      for v in attribution["delta_vs_lastclick"]]
            fig3 = go.Figure(go.Bar(
                x=attribution["name"],
                y=attribution["delta_vs_lastclick"] * 100,
                marker_color=colors, marker_line_width=0,
            ))
            fig3.add_hline(y=0, line_color=BORDER, line_width=1.5)
            chart_layout(fig3, 240)
            fig3.update_layout(yaxis_title="Δ Credit (pp)")
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Full attribution table</div>', unsafe_allow_html=True)
        disp = attribution[["name","removal_effect","shapley_credit",
                             "last_click_share","delta_vs_lastclick"]].copy()
        disp.columns = ["Creator","Removal Effect","Shapley Credit",
                        "Last-Click Share","Δ vs Last-Click"]
        for c in disp.columns[1:]:
            disp[c] = disp[c].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif page == "Sentiment Analysis":
    st.markdown('<div class="page-title">Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">BERT-powered analysis of 1,352 real YouTube comments — measuring booking intent, not just positive vibes.</div>', unsafe_allow_html=True)

    if sentiment.empty:
        st.warning("Run step3_sentiment.py first.")
    else:
        cols = st.columns(len(sentiment))
        for i, (_, row) in enumerate(sentiment.iterrows()):
            with cols[i]:
                sc   = row.get("avg_intent_score", 0)
                pos  = row.get("positive_rate", 0) * 100
                bi   = row.get("booking_intent_rate", 0) * 100
                n    = row.get("name", row.get("creator",""))
                tc   = int(row.get("total_comments", 0))
                # progress bar width — normalise to max score
                max_sc = sentiment["avg_intent_score"].max() or 1
                bar_w = min(sc / max_sc * 100, 100)
                st.markdown(f"""
                <div class="at-card-top">
                    <div class="stat-label">{n}</div>
                    <div class="stat-number">{sc:.2f}</div>
                    <div style='background: #f1f5f9; border-radius: 4px;
                                height: 5px; overflow: hidden; margin: 10px 0;'>
                        <div style='height: 5px; border-radius: 4px;
                                    width: {bar_w:.0f}%;
                                    background: {ACCENT};'></div>
                    </div>
                    <div class="stat-sub">{pos:.0f}% positive sentiment</div>
                    <div class="stat-sub">{bi:.2f}% booking intent</div>
                    <div class="stat-sub">{tc:,} comments</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cl, cr = st.columns(2)

        with cl:
            st.markdown('<div class="section-label">Positive sentiment rate</div>', unsafe_allow_html=True)
            fig = px.bar(sentiment.sort_values("positive_rate", ascending=True),
                         x="positive_rate", y="name", orientation="h",
                         color="positive_rate", color_continuous_scale=SCALE,
                         labels={"positive_rate": "Positive Rate", "name": ""})
            chart_layout(fig, 240)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.markdown('<div class="section-label">Average intent score</div>', unsafe_allow_html=True)
            fig2 = px.bar(sentiment.sort_values("avg_intent_score", ascending=True),
                          x="avg_intent_score", y="name", orientation="h",
                          color="avg_intent_score", color_continuous_scale=SCALE,
                          labels={"avg_intent_score": "Avg Intent Score", "name": ""})
            chart_layout(fig2, 240)
            fig2.update_layout(coloraxis_showscale=False)
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-label">How the intent score is calculated</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        scoring = [
            ("+2", "Booking intent phrase",
             '"where is this hotel?", "just booked", "adding to my list"'),
            ("+1", "Positive BERT sentiment",
             "RoBERTa model classifies comment as positive"),
            ("−1", "Negative sentiment",
             "Negative audience reaction reduces the creator score"),
        ]
        for col, (num, title, desc) in zip([c1,c2,c3], scoring):
            with col:
                st.markdown(f"""
                <div class="at-card" style='text-align: center;
                             border-top: 3px solid {ACCENT};'>
                    <div style='font-family: Syne, sans-serif; font-size: 36px;
                                font-weight: 800; color: {ACCENT};'>{num}</div>
                    <div style='font-weight: 600; font-size: 13px;
                                color: {TEXT}; margin: 8px 0 5px;'>{title}</div>
                    <div style='font-size: 12px; color: {MUTED};
                                line-height: 1.55;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAYOUT CALCULATOR
# ══════════════════════════════════════════════════════════════════
elif page == "Payout Calculator":
    st.markdown('<div class="page-title">Payout Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">How much should each creator earn? Fair payout based on Shapley attribution — not follower count or guesswork.</div>', unsafe_allow_html=True)

    if attribution.empty:
        st.warning("Run step5_train_model.py first.")
    else:
        st.markdown('<div class="section-label">Campaign settings</div>', unsafe_allow_html=True)
        ca, cb, cc = st.columns(3)
        with ca:
            budget = st.number_input("Total campaign budget (€)",
                                     min_value=1000, max_value=1000000,
                                     value=10000, step=500)
        with cb:
            model_choice = st.selectbox("Attribution model", [
                "Shapley values (recommended)",
                "Markov removal effect",
                "Last-click (for comparison)",
            ])
        with cc:
            floor = st.slider("Minimum payout floor (€)", 0, 1000, 100, 50)

        st.markdown("<br>", unsafe_allow_html=True)

        # compute payouts
        if "Shapley" in model_choice:
            w = attribution["shapley_credit"].values
            mlabel = "Shapley"
        elif "Markov" in model_choice:
            w = attribution["removal_effect"].values
            w = w / w.sum() if w.sum() > 0 else w
            mlabel = "Markov"
        else:
            w = attribution["last_click_share"].values
            mlabel = "Last-click"

        raw     = w * budget
        payouts = np.maximum(raw, floor)
        if payouts.sum() > 0:
            payouts = payouts / payouts.sum() * budget

        payout_df = attribution.copy()
        payout_df["payout"]    = payouts
        payout_df["payout_pct"] = payouts / budget * 100
        payout_df["credit"]    = w * 100

        # payout cards
        st.markdown('<div class="section-label">Recommended payouts</div>', unsafe_allow_html=True)
        cols = st.columns(len(payout_df))
        for i, (_, row) in enumerate(payout_df.iterrows()):
            with cols[i]:
                pct = row["payout_pct"]
                bar_w = min(pct * 3, 100)
                st.markdown(f"""
                <div class="at-card-top">
                    <div class="stat-label">{row['name']}</div>
                    <div class="stat-number">€{row['payout']:,.0f}</div>
                    <div style='background: #f1f5f9; border-radius: 4px;
                                height: 5px; overflow: hidden; margin: 10px 0;'>
                        <div style='height: 5px; border-radius: 4px;
                                    width: {bar_w:.0f}%;
                                    background: {ACCENT};'></div>
                    </div>
                    <div class="stat-sub">{pct:.1f}% of budget</div>
                    <div class="stat-sub">{row['credit']:.1f}% attribution credit</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Fair payout vs last-click comparison</div>', unsafe_allow_html=True)

        lc_pay = attribution["last_click_share"] * budget
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Last-click payout",
            x=payout_df["name"], y=lc_pay,
            marker_color="#e2e8f0", marker_line_width=0,
        ))
        fig.add_trace(go.Bar(
            name=f"{mlabel} payout",
            x=payout_df["name"], y=payout_df["payout"],
            marker_color=ACCENT, marker_line_width=0,
        ))
        chart_layout(fig, 300)
        fig.update_layout(
            barmode="group",
            yaxis_title="Payout (€)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Full payout breakdown</div>', unsafe_allow_html=True)
        t = payout_df[["name","credit","payout","payout_pct"]].copy()
        t.columns = ["Creator","Attribution Credit (%)","Payout (€)","% of Budget"]
        t["Attribution Credit (%)"] = t["Attribution Credit (%)"].apply(lambda x: f"{x:.2f}%")
        t["Payout (€)"]             = t["Payout (€)"].apply(lambda x: f"€{x:,.2f}")
        t["% of Budget"]            = t["% of Budget"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(t, use_container_width=True, hide_index=True)

        total = payout_df["payout"].sum()
        st.markdown(f"""
        <div class="at-card" style='display: flex; justify-content: space-between;
                                     align-items: center; border-top: 3px solid {ACCENT};'>
            <div>
                <div class="stat-label">Total budget allocated</div>
                <div style='font-family: Syne, sans-serif; font-size: 26px;
                            font-weight: 800; color: {TEXT};'>
                    €{total:,.2f}
                </div>
            </div>
            <div style='text-align: right;'>
                <div class="stat-label">Model</div>
                <div style='font-size: 18px; font-weight: 700; color: {ACCENT};'>
                    {mlabel}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)