from __future__ import annotations

import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_collection import DataCollector
from preprocessing import DataPreprocessor
from sentiment_emotion import SentimentEmotionAnalyzer
from user_profiling import UserProfiler
from recommendation_model import (
    CollaborativeFiltering, ContentBasedFiltering, HybridRecommender,
)
from ranking import RankingEngine
from recommendation_delivery import RecommendationDelivery
from feedback_loop import FeedbackLoop
from database import RecSysDatabase
from evaluation import Evaluator
from content_sources import CuratedContentProvider


# --------------------------------------------------------------------------- #
# Page config & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Personalised Motivational",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORY_COLORS = {
    "confidence_building": "#7C5CFC",
    "stress_relief": "#3EC1D3",
    "career_motivation": "#FF9F43",
    "physical_fitness": "#28C76F",
    "academic_motivation": "#EA5455",
    "mindfulness": "#00CFE8",
    "leadership": "#7367F0",
    "relationships": "#FF6B9A",
}
TYPE_ICON = {"video": "🎬", "podcast": "🎧", "speech": "🎤", "quote": "💬", "article": "📰"}
EMOTION_EMOJI = {
    "stressed": "😰", "sad": "😔", "angry": "😠",
    "confident": "😎", "excited": "🤩", "neutral": "🙂",
}

CUSTOM_CSS = """
<style>
.main .block-container {padding-top: 2rem; max-width: 1200px;}
.hero {
    background: linear-gradient(135deg, #7C5CFC 0%, #3EC1D3 100%);
    padding: 28px 32px; border-radius: 16px; color: white; margin-bottom: 24px;
}
.hero h1 {margin: 0; font-size: 28px;}
.hero p {margin: 4px 0 0 0; opacity: 0.9;}
.rec-card {
    border-radius: 14px; padding: 18px; background: #ffffff;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07); border: 1px solid #eef0f4;
    margin-bottom: 14px; transition: transform 0.15s ease;
}
.rec-card:hover {transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.10);}
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600; color: white; margin-right: 6px;
}
.score-pill {
    background: #F1F0FF; color: #6C4CF1; padding: 3px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 700; float: right;
}
.explain {color: #6b7280; font-size: 13px; margin-top: 6px;}
.metric-card {
    background: white; border-radius: 12px; padding: 16px; text-align: center;
    border: 1px solid #eef0f4; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading content & training recommendation models...")
def bootstrap_pipeline(n_users=200, n_interactions=4000):
    collector = DataCollector()

    # --- Block 1: Data Collection -> CURATED media catalog -----------------
    media_raw = CuratedContentProvider().build_media_catalog()

    users_raw = collector.generate_users(n_users)
    interactions_raw = collector.generate_interactions(users_raw, media_raw, n_interactions)
    raw = {"users": users_raw, "media": media_raw, "interactions": interactions_raw}

    pre = DataPreprocessor()
    users, media, interactions = pre.clean_dataframes(**raw)
    tfidf_matrix = pre.build_media_text_features(media)
    interactions = pre.build_interaction_features(interactions)
    splits = pre.train_val_test_split(interactions)

    profiler = UserProfiler(categories=DataCollector.CATEGORIES)
    profiles = profiler.build_profiles(users, media, interactions)

    cf = CollaborativeFiltering(n_factors=20).fit(splits["train"])
    cbf = ContentBasedFiltering(tfidf_matrix, pre.media_ids_order)
    hybrid = HybridRecommender(cf, cbf, cf_weight=0.55)

    ranker = RankingEngine()
    ranker.fit_engagement_model(splits["train"], media)

    db = RecSysDatabase(":memory:")
    db.save_users(users)
    db.save_media(media)
    db.save_interactions(interactions)

    return {
        "users": users, "media": media, "interactions": interactions,
        "profiles": profiles, "splits": splits, "cf": cf, "cbf": cbf,
        "hybrid": hybrid, "ranker": ranker, "db": db,
    }


@st.cache_resource
def get_delivery():
    return RecommendationDelivery(media_type_icons=TYPE_ICON)


@st.cache_resource
def get_analyzer():
    return SentimentEmotionAnalyzer()


if "feedback_loop" not in st.session_state:
    st.session_state.feedback_loop = FeedbackLoop()
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "last_payload" not in st.session_state:
    st.session_state.last_payload = []
if "active_user" not in st.session_state:
    st.session_state.active_user = None


delivery = get_delivery()
analyzer = get_analyzer()


with st.sidebar:
    st.markdown("  🌐 Content")
 

    st.markdown("---")
    ctx = bootstrap_pipeline()

    st.markdown(" 👤 User Profile")
    user_ids = ctx["users"]["user_id"].tolist()
    selected_user = st.selectbox("Select user", user_ids, index=0)

    # Streamlit reruns the script when the user changes. Clear cached UI
    # outputs so the previous user's recommendations/evaluation are not shown.
    if st.session_state.active_user != selected_user:
        st.session_state.active_user = selected_user
        st.session_state.last_payload = []
        st.session_state.pop("last_sentiment", None)
        st.session_state.pop("report_df", None)

    user_row = ctx["users"].set_index("user_id").loc[selected_user]
    profile_row = ctx["profiles"].set_index("user_id").loc[selected_user]

    st.write(f"**Age:** {user_row['age']}  |  **Profession:** {user_row['profession']}")
    st.write(f"**Interests:** {', '.join(user_row['interests'])}")
    st.write(f"**Top affinity category:** `{profile_row['top_category']}`")

    st.markdown("---")
    st.markdown(" 💭 How are you feeling?")
    mood_text = st.text_area(
        "Tell us what's on your mind (used for sentiment & emotion analysis)",
        placeholder="e.g. I feel stressed and overwhelmed with my exams...",
        height=90,
    )
    top_n = st.slider("Number of recommendations", min_value=3, max_value=12, value=6)
    get_recs = st.button("Get Recommendations", use_container_width=True, type="primary")

    
    
st.markdown(
    """
    <div class="hero">
        <h1> Personalized Motivational Media Recommendation System Using ML</h1>
        
    </div>
    """,
    unsafe_allow_html=True,
)

tab_recs, tab_analytics, tab_architecture = st.tabs(
    ["🎯 Recommendations", "📊 Model Analytics", "🏗️ Dataset"]
)


# --------------------------------------------------------------------------- #
# TAB 1: Recommendations (Blocks 3, 4, 5)
# --------------------------------------------------------------------------- #
with tab_recs:
    if get_recs or st.session_state.last_payload:
        if get_recs:
            with st.spinner("Analyzing mood, ranking content..."):
                sentiment_result = analyzer.analyze(mood_text)
                bias_category = analyzer.category_for_emotion(sentiment_result.emotion)

                already_seen = ctx["interactions"].loc[
                    ctx["interactions"]["user_id"] == selected_user, "media_id"
                ].tolist()

                candidates = ctx["hybrid"].recommend(
                    selected_user, ctx["splits"]["train"], ctx["media"], top_n=50,
                    exclude=already_seen, bias_category=bias_category,
                )
                ranked = ctx["ranker"].rank(
                    candidates, ctx["splits"]["train"], ctx["media"], top_n=top_n
                )
                payload = delivery.deliver(
                    ranked, selected_user, current_emotion=sentiment_result.emotion,
                    bias_category=bias_category,
                )
                st.session_state.last_payload = payload
                st.session_state.last_sentiment = sentiment_result

        payload = st.session_state.last_payload
        sentiment_result = st.session_state.get("last_sentiment")

        if sentiment_result is not None:
            emoji = EMOTION_EMOJI.get(sentiment_result.emotion, "🙂")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-card'><h3>{emoji}</h3>"
                        f"<b>Detected emotion</b><br>{sentiment_result.emotion.title()}</div>",
                        unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><h3>{sentiment_result.sentiment}</h3>"
                        f"<b>Sentiment</b><br>compound={sentiment_result.compound:+.2f}</div>",
                        unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><h3>{len(payload)}</h3>"
                        f"<b>Recommendations delivered</b></div>", unsafe_allow_html=True)
            st.write("")

        cols = st.columns(2)
        for idx, item in enumerate(payload):
            col = cols[idx % 2]
            color = CATEGORY_COLORS.get(item["category"], "#7C5CFC")
            with col:
                title_html = (
                    f'<a href="{item["url"]}" target="_blank" style="color:#1f2937;text-decoration:none;">'
                    f'{item["icon"]} {item["title"]} 🔗</a>'
                    if item.get("url") else f'{item["icon"]} {item["title"]}'
                )
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <span class="score-pill">score {item['score']:.2f}</span>
                        <span class="badge" style="background:{color}">
                            {item['category'].replace('_',' ').title()}
                        </span>
                        <h4 style="margin:10px 0 2px 0;">{title_html}</h4>
                        <div class="explain">{item['explanation']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fb1, fb2, fb3 = st.columns(3)
                like_key = f"like_{item['media_id']}_{idx}"
                skip_key = f"skip_{item['media_id']}_{idx}"
                share_key = f"share_{item['media_id']}_{idx}"

                if fb1.button("👍 Like", key=like_key, use_container_width=True):
                    st.session_state.feedback_loop.record_feedback(
                        selected_user, item["media_id"], liked=True, rating=5,
                        watch_time_pct=0.95,
                    )
                    st.session_state.feedback_log.append((item["title"], "Liked"))
                    st.toast(f"Feedback recorded: liked '{item['title']}'", icon="👍")

                if fb2.button("👎 Skip", key=skip_key, use_container_width=True):
                    st.session_state.feedback_loop.record_feedback(
                        selected_user, item["media_id"], liked=False, rating=2,
                        watch_time_pct=0.1, skipped=True,
                    )
                    st.session_state.feedback_log.append((item["title"], "Skipped"))
                    st.toast(f"Feedback recorded: skipped '{item['title']}'", icon="👎")

                if fb3.button("🔗 Share", key=share_key, use_container_width=True):
                    st.session_state.feedback_loop.record_feedback(
                        selected_user, item["media_id"], liked=True, rating=5,
                        watch_time_pct=1.0, shared=True,
                    )
                    st.session_state.feedback_log.append((item["title"], "Shared"))
                    st.toast(f"Shared '{item['title']}'!", icon="🔗")

        if st.session_state.feedback_log:
            st.markdown("---")
            st.markdown("Feedback captured this session (Block: Feedback Loop)")
            fb_df = pd.DataFrame(st.session_state.feedback_log, columns=["Title", "Action"])
            st.dataframe(fb_df, use_container_width=True, hide_index=True)
    else:
        st.info(" Select a user, describe your mood (optional), and click "
                "**Get Recommendations** to see personalized motivational media.")


# --------------------------------------------------------------------------- #
# TAB 2: Model Analytics (Block: Performance Evaluation)
# --------------------------------------------------------------------------- #
with tab_analytics:
    st.markdown(f"Model comparison for selected user: **{selected_user}**")
    st.caption("Metrics are calculated against this user's held-out relevant items.")

    eval_signature = (selected_user, top_n)
    force_eval = st.button("Run evaluation", type="secondary", key="run_selected_user_eval")
    if force_eval or st.session_state.get("eval_signature") != eval_signature:
        with st.spinner(f"Evaluating recommenders for {selected_user}..."):
            evaluator = Evaluator(k=top_n)
            train_seen = ctx["splits"]["train"].loc[
                ctx["splits"]["train"]["user_id"] == selected_user, "media_id"
            ].tolist()

            def hybrid_fn(uid, top_n=10):
                return ctx["hybrid"].recommend(
                    uid, ctx["splits"]["train"], ctx["media"],
                    top_n=top_n, exclude=train_seen
                )

            def cf_fn(uid, top_n=10):
                return ctx["cf"].recommend(uid, top_n=top_n, exclude=train_seen)

            def cbf_fn(uid, top_n=10):
                return ctx["cbf"].recommend(
                    uid, ctx["splits"]["train"], top_n=top_n, exclude=train_seen
                )

            recommenders = {
                "Collaborative Filtering": cf_fn,
                "Content-Based Filtering": cbf_fn,
                "Hybrid (proposed)": hybrid_fn,
            }

            user_results = evaluator.evaluate_user_models(
                recommenders, selected_user, ctx["splits"]["test"], k=top_n
            )

            if user_results.empty:
                st.session_state.report_df = pd.DataFrame()
                st.session_state.eval_user = selected_user
                st.session_state.eval_signature = eval_signature
                st.info(
                    f"{selected_user} has no relevant held-out test items "
                    "(rating >= 4 or liked=True), so ranking metrics cannot be computed."
                )
            else:
                report_df = user_results.set_index("model")[[
                    "precision", "recall", "f1", "ndcg"
                ]].copy()
                report_df.columns = [str(col).replace("@k", "") for col in report_df.columns]
                st.session_state.report_df = report_df
                st.session_state.eval_user = selected_user
                st.session_state.eval_signature = eval_signature

    if "report_df" in st.session_state and st.session_state.get("eval_user") == selected_user:
        report_df = st.session_state.report_df
        if report_df.empty:
            st.info("Run evaluation after selecting another user with held-out relevant items.")
        else:
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

            fig = go.Figure()
            for metric in report_df.columns:
                fig.add_trace(go.Bar(name=metric, x=report_df.index, y=report_df[metric]))
            fig.update_layout(
                barmode="group", height=420,
                title=f"Recommender comparison for {selected_user}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click **Run evaluation** to evaluate the currently selected user.")

    # Evaluate the exact recommendations currently visible in Tab 1.
    if st.session_state.get("last_payload"):
        st.markdown("#### Current recommendation evaluation")
        evaluator = Evaluator(k=len(st.session_state.last_payload))
        current_ids = [item["media_id"] for item in st.session_state.last_payload]
        current_metrics = evaluator.evaluate_recommendation_list(
            selected_user, current_ids, ctx["splits"]["test"],
            k=len(current_ids)
        )

        if current_metrics["evaluated"]:
            current_df = pd.DataFrame([{
                "user": selected_user,
                "precision": current_metrics["precision"],
                "recall": current_metrics["recall"],
                "f1": current_metrics["f1"],
                "ndcg": current_metrics["ndcg"],
                "relevant test items": current_metrics["relevant_count"],
            }]).set_index("user")
            st.dataframe(current_df.style.format({
                "precision": "{:.4f}", "recall": "{:.4f}",
                "f1": "{:.4f}", "ndcg": "{:.4f}",
            }), use_container_width=True)
            st.caption("These values are based on the exact recommendation list currently displayed for this user.")
        else:
            st.info(f"No held-out relevant items are available for {selected_user} yet.")

    st.markdown("---")
    st.markdown("#### Category distribution in the media catalog")
    cat_counts = ctx["media"]["category"].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]
    fig2 = px.bar(cat_counts, x="category", y="count", color="category",
                  color_discrete_map=CATEGORY_COLORS, template="plotly_white")
    fig2.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig2, use_container_width=True)


 
# TAB 3: Architecture overview
 
with tab_architecture:
    

    with st.expander("Dataset Details"):
        c1, c2 = st.columns(2)
        c1.markdown("**Users**")
        c1.dataframe(ctx["users"].head(10), use_container_width=True, hide_index=True)
        c2.markdown("**Media catalog**")
        c2.dataframe(ctx["media"].head(10), use_container_width=True, hide_index=True)