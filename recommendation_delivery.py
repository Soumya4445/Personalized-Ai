
from __future__ import annotations

from typing import List, Dict
import pandas as pd


class RecommendationDelivery:
    def __init__(self, media_type_icons: Dict[str, str] = None):
        self.icons = media_type_icons or {
            "video": "🎬", "podcast": "🎧", "speech": "🎤",
            "quote": "💬", "article": "📰",
        }

    def _explain(self, row: pd.Series, current_emotion: str,
                 bias_category: str) -> str:
        reasons = []
        if row["category"] == bias_category:
            reasons.append(f"matches your current mood ({current_emotion})")
        if row["popularity"] > 0.6:
            reasons.append("popular with similar users")
        if row["engagement_probability"] > 0.6:
            reasons.append("likely to keep you engaged")
        if not reasons:
            reasons.append("similar to content you've enjoyed before")
        return "Recommended because it " + " and ".join(reasons) + "."

    def deliver(self, ranked_df: pd.DataFrame, user_id: str,
                current_emotion: str = "neutral",
                bias_category: str = "") -> List[Dict]:
        payload = []
        for _, row in ranked_df.iterrows():
            payload.append({
                "user_id": user_id,
                "media_id": row["media_id"],
                "title": row["title"],
                "type": row["media_type"],
                "icon": self.icons.get(row["media_type"], "🔹"),
                "category": row["category"],
                "score": round(float(row["final_score"]), 4),
                "explanation": self._explain(row, current_emotion, bias_category),
                "url": row["url"] if "url" in row and pd.notna(row["url"]) else "",
            })
        return payload

    @staticmethod
    def pretty_print(payload: List[Dict]):
        print(f"\nTop {len(payload)} Recommendations")
        print("-" * 70)
        for i, item in enumerate(payload, 1):
            print(f"{i}. {item['icon']} {item['title']}  (score={item['score']})")
            print(f"   {item['explanation']}")
            if item.get("url"):
                print(f"   🔗 {item['url']}")


if __name__ == "__main__":
    from data_collection import DataCollector
    from preprocessing import DataPreprocessor
    from recommendation_model import CollaborativeFiltering, ContentBasedFiltering, HybridRecommender
    from ranking import RankingEngine

    raw = DataCollector().collect_all(n_users=50, n_media=150, n_interactions=1500)
    pre = DataPreprocessor()
    users, media, interactions = pre.clean_dataframes(**raw)
    tfidf_matrix = pre.build_media_text_features(media)
    feats = pre.build_interaction_features(interactions)

    cf = CollaborativeFiltering(n_factors=15).fit(feats)
    cbf = ContentBasedFiltering(tfidf_matrix, pre.media_ids_order)
    hybrid = HybridRecommender(cf, cbf)

    test_user = users.iloc[0]["user_id"]
    candidates = hybrid.recommend(test_user, feats, media, top_n=30, bias_category="stress_relief")
    ranked = RankingEngine().rank(candidates, feats, media, top_n=5)

    delivery = RecommendationDelivery()
    payload = delivery.deliver(ranked, test_user, current_emotion="stressed",
                                bias_category="stress_relief")
    delivery.pretty_print(payload)
