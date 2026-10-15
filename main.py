

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

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


class MotivationalRecommenderPipeline:
    """High-level facade that runs the full architecture end-to-end."""

    def __init__(self, n_users=200, n_media=500, n_interactions=4000, db_path=":memory:"):
        self.n_users, self.n_media, self.n_interactions = n_users, n_media, n_interactions
        self.db = RecSysDatabase(db_path)

        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.sentiment_analyzer = SentimentEmotionAnalyzer()
        self.profiler = UserProfiler(categories=DataCollector.CATEGORIES)
        self.feedback_loop = FeedbackLoop()
        self.evaluator = Evaluator(k=10)

        self.users = self.media = self.interactions = None
        self.tfidf_matrix = None
        self.splits = None
        self.cf = self.cbf = self.hybrid = None
        self.ranker = RankingEngine()
        self.delivery = RecommendationDelivery()

    # ------------------------------------------------------------------ #
    def run_data_pipeline(self, use_real_content: bool = True):
        print("STEP 1-2 | Collecting & preprocessing data ...")
        if use_real_content:
            media_raw = CuratedContentProvider().build_media_catalog()
            print(f"    Using REAL curated content catalog ({len(media_raw)} real items).")
        else:
            media_raw = self.collector.generate_media(self.n_media)

        users_raw = self.collector.generate_users(self.n_users)
        interactions_raw = self.collector.generate_interactions(
            users_raw, media_raw, self.n_interactions
        )
        raw = {"users": users_raw, "media": media_raw, "interactions": interactions_raw}
        self.users, self.media, self.interactions = self.preprocessor.clean_dataframes(**raw)
        self.tfidf_matrix = self.preprocessor.build_media_text_features(self.media)
        self.interactions = self.preprocessor.build_interaction_features(self.interactions)
        self.splits = self.preprocessor.train_val_test_split(self.interactions)

        self.db.save_users(self.users)
        self.db.save_media(self.media)
        self.db.save_interactions(self.interactions)
        print(f"    users={self.users.shape[0]} media={self.media.shape[0]} "
              f"interactions={self.interactions.shape[0]}")

    def build_profiles(self):
        print("STEP 3-4 | Sentiment/emotion analysis & user profiling ...")
        self.profiles = self.profiler.build_profiles(self.users, self.media, self.interactions)

    def train_models(self):
        print("STEP 5   | Training recommendation models (CF, CBF, Hybrid) ...")
        self.cf = CollaborativeFiltering(n_factors=20).fit(self.splits["train"])
        self.cbf = ContentBasedFiltering(self.tfidf_matrix, self.preprocessor.media_ids_order)
        self.hybrid = HybridRecommender(self.cf, self.cbf, cf_weight=0.55)
        self.ranker.fit_engagement_model(self.splits["train"], self.media)
        self.db.log_model_training("hybrid_recommender_v1",
                                    {"n_factors": 20, "cf_weight": 0.55})

    # ------------------------------------------------------------------ #
    def recommend_for_user(self, user_id: str, top_n: int = 10, mood_text: str = ""):
        print(f"\nSTEP 6-7 | Ranking & delivering Top-{top_n} for {user_id} ...")

        sentiment_result = self.sentiment_analyzer.analyze(mood_text)
        bias_category = self.sentiment_analyzer.category_for_emotion(sentiment_result.emotion)

        already_seen = self.interactions.loc[
            self.interactions["user_id"] == user_id, "media_id"
        ].tolist()

        candidates = self.hybrid.recommend(
            user_id, self.splits["train"], self.media, top_n=50,
            exclude=already_seen, bias_category=bias_category,
        )
        ranked = self.ranker.rank(candidates, self.splits["train"], self.media, top_n=top_n)
        payload = self.delivery.deliver(
            ranked, user_id, current_emotion=sentiment_result.emotion,
            bias_category=bias_category,
        )
        self.delivery.pretty_print(payload)
        return payload

    # ------------------------------------------------------------------ #
    def simulate_feedback(self, payload, accept_rate: float = 0.6):
        print("\nSTEP 8   | Simulating user feedback on delivered recommendations ...")
        import random
        for item in payload:
            liked = random.random() < accept_rate
            self.feedback_loop.record_feedback(
                user_id=item["user_id"], media_id=item["media_id"],
                liked=liked, rating=(5 if liked else 2),
                watch_time_pct=(0.9 if liked else 0.15), skipped=not liked,
            )
        new_feedback_df = self.feedback_loop.as_dataframe()
        self.db.save_interactions(new_feedback_df)
        print(f"    Recorded {len(new_feedback_df)} feedback events.")

    # ------------------------------------------------------------------ #
    def evaluate(self):
        print("\nSTEP 10  | Performance evaluation (Precision/Recall/F1/NDCG) ...")

        def hybrid_fn(uid, top_n=10):
            return self.hybrid.recommend(uid, self.splits["train"], self.media, top_n=top_n)

        def cf_fn(uid, top_n=10):
            return self.cf.recommend(uid, top_n=top_n)

        def cbf_fn(uid, top_n=10):
            return self.cbf.recommend(uid, self.splits["train"], top_n=top_n)

        test_users = self.splits["test"]["user_id"].unique().tolist()[:40]
        report = {}
        for name, fn in [("Collaborative Filtering", cf_fn),
                          ("Content-Based Filtering", cbf_fn),
                          ("Hybrid (proposed)", hybrid_fn)]:
            res = self.evaluator.evaluate_ranking(fn, self.splits["test"], sample_users=test_users)
            report[name] = self.evaluator.summarize(res)

        import pandas as pd
        report_df = pd.DataFrame(report).T
        print(report_df)
        return report_df


def run_demo():
    pipeline = MotivationalRecommenderPipeline(n_users=200, n_media=500, n_interactions=4000)
    pipeline.run_data_pipeline()
    pipeline.build_profiles()
    pipeline.train_models()

    demo_user = pipeline.users.iloc[0]["user_id"]
    payload = pipeline.recommend_for_user(
        demo_user, top_n=5, mood_text="I feel stressed and overwhelmed with my exams"
    )
    pipeline.simulate_feedback(payload)
    pipeline.evaluate()


if __name__ == "__main__":
    run_demo()
