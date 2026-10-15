

from __future__ import annotations

import re
import string
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# A small built-in stop-word list keeps this module dependency-free (no NLTK
# corpus download required). Swap in nltk.corpus.stopwords / spaCy in
# production if richer linguistic processing is needed.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "for", "with", "at", "by",
    "this", "that", "it", "as", "i", "my", "me", "so", "very", "just",
}


def clean_text(text: str) -> str:
    """Lower-case, strip punctuation/digits, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_stem(word: str) -> str:
    """Very small suffix-stripping stemmer (placeholder for Porter/Snowball)."""
    for suffix in ("ing", "edly", "edness", "ed", "ness", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[: -len(suffix)]
    return word


def tokenize_and_normalize(text: str) -> str:
    """Tokenize, remove stop-words, stem -> return normalized string."""
    cleaned = clean_text(text)
    tokens = [t for t in cleaned.split() if t not in _STOPWORDS and len(t) > 1]
    stemmed = [simple_stem(t) for t in tokens]
    return " ".join(stemmed)


class DataPreprocessor:
    """
    Cleans raw dataframes, builds TF-IDF text features for media content,
    normalizes numeric interaction features, and produces train/val/test
    splits of the interaction table used to train/evaluate the recommender.
    """

    def __init__(self, tfidf_max_features: int = 300):
        self.tfidf = TfidfVectorizer(max_features=tfidf_max_features)
        self.scaler = MinMaxScaler()
        self.category_encoder = LabelEncoder()
        self.media_tfidf_matrix = None
        self.media_ids_order = None

    # ------------------------------------------------------------------ #
    def clean_dataframes(self, users: pd.DataFrame, media: pd.DataFrame,
                          interactions: pd.DataFrame
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        users = users.drop_duplicates(subset="user_id").reset_index(drop=True)
        media = media.drop_duplicates(subset="media_id").reset_index(drop=True)
        interactions = interactions.drop_duplicates().reset_index(drop=True)

        # Remove noisy / impossible rows
        interactions = interactions[interactions["watch_time_pct"].between(0, 1)]
        interactions["rating"] = interactions["rating"].astype("float")
        # Impute missing ratings with an implicit estimate derived from
        # engagement signals rather than dropping rows (keeps sparsity low)
        implicit_rating = (
            1 + 4 * interactions["watch_time_pct"].fillna(0)
        ).clip(1, 5)
        interactions["rating_filled"] = interactions["rating"].fillna(implicit_rating)
        interactions["comment"] = interactions["comment"].fillna("")
        interactions["mood_input"] = interactions["mood_input"].fillna("")
        return users, media, interactions.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def build_media_text_features(self, media: pd.DataFrame) -> np.ndarray:
        """TF-IDF representation of media content (title + description + tags)."""
        corpus = (
            media["title"] + " " + media["description"] + " " +
            media["tags"].apply(lambda t: " ".join(t) if isinstance(t, list) else str(t)) +
            " " + media["category"]
        ).apply(tokenize_and_normalize)

        self.media_tfidf_matrix = self.tfidf.fit_transform(corpus)
        self.media_ids_order = media["media_id"].tolist()
        return self.media_tfidf_matrix

    # ------------------------------------------------------------------ #
    def build_interaction_features(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric behavioral features used downstream for scoring."""
        df = interactions.copy()
        numeric_cols = ["rating_filled", "watch_time_pct"]
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        df["liked_flag"] = df["liked"].astype(int)
        df["skipped_flag"] = df["skipped"].astype(int)
        return df

    # ------------------------------------------------------------------ #
    @staticmethod
    def train_val_test_split(interactions: pd.DataFrame, val_size=0.15, test_size=0.15,
                              random_state=42) -> Dict[str, pd.DataFrame]:
        train, temp = train_test_split(
            interactions, test_size=(val_size + test_size), random_state=random_state
        )
        rel_test = test_size / (val_size + test_size)
        val, test = train_test_split(temp, test_size=rel_test, random_state=random_state)
        return {"train": train.reset_index(drop=True),
                "val": val.reset_index(drop=True),
                "test": test.reset_index(drop=True)}


if __name__ == "__main__":
    from data_collection import DataCollector

    raw = DataCollector().collect_all()
    pre = DataPreprocessor()
    users, media, interactions = pre.clean_dataframes(**raw)
    tfidf_matrix = pre.build_media_text_features(media)
    feats = pre.build_interaction_features(interactions)
    splits = pre.train_val_test_split(feats)

    print("TF-IDF media matrix shape:", tfidf_matrix.shape)
    print({k: v.shape for k, v in splits.items()})
