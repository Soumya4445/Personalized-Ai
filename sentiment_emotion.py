
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class SentimentResult:
    text: str
    compound: float
    sentiment: str      # Positive / Negative / Neutral
    emotion: str         # Sad / Stressed / Angry / Confident / Excited / Neutral


class SentimentEmotionAnalyzer:
    """Wraps VADER sentiment scoring with a simple emotion lexicon layer."""

    EMOTION_KEYWORDS: Dict[str, List[str]] = {
        "stressed": ["stress", "overwhelm", "anxious", "anxiety", "pressure", "exam", "deadline"],
        "sad": ["sad", "down", "low", "unmotivated", "depressed", "lonely", "tired"],
        "angry": ["angry", "frustrat", "annoyed", "mad", "furious"],
        "confident": ["confident", "ready", "strong", "capable", "proud"],
        "excited": ["excited", "thrilled", "pumped", "motivated", "great", "amazing"],
    }

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def _detect_emotion(self, text: str, compound: float) -> str:
        text_lower = text.lower()
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return emotion
        # Fallback purely from polarity score if no keyword hit
        if compound >= 0.5:
            return "excited"
        if compound <= -0.5:
            return "sad"
        return "neutral"

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(text=text, compound=0.0, sentiment="Neutral", emotion="neutral")

        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        emotion = self._detect_emotion(text, compound)
        return SentimentResult(text=text, compound=compound, sentiment=sentiment, emotion=emotion)

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        return [self.analyze(t) for t in texts]

    def category_for_emotion(self, emotion: str) -> str:
        """Maps a detected emotion to a recommended motivational media category
        (used later by the recommendation engine to bias suggestions)."""
        mapping = {
            "stressed": "stress_relief",
            "sad": "mindfulness",
            "angry": "mindfulness",
            "confident": "career_motivation",
            "excited": "physical_fitness",
            "neutral": "confidence_building",
        }
        return mapping.get(emotion, "confidence_building")


if __name__ == "__main__":
    analyzer = SentimentEmotionAnalyzer()
    samples = [
        "I feel stressed and overwhelmed with my exams",
        "This video really boosted my confidence, loved it!",
        "Feeling low on energy and unmotivated today",
        "",
    ]
    for s in samples:
        r = analyzer.analyze(s)
        print(f"{r.sentiment:8s} | {r.emotion:10s} | compound={r.compound:+.2f} | {s!r}")
