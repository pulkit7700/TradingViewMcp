"""
sentiment_engine.py
===================
Financial news sentiment analysis using FinBERT (primary) or VADER (fallback).

FinBERT: ProsusAI/finbert — BERT fine-tuned on financial text.
VADER: Rule-based lexicon, lightweight, no GPU required.

Usage
-----
engine = SentimentEngine()
result = engine.analyze_ticker("AAPL", news_items)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings
import time

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"
MAX_TOKEN_LENGTH = 512
BATCH_SIZE = 8
STRONG_THRESHOLD = 0.4     # |score| > 0.4 = strong signal
MODERATE_THRESHOLD = 0.15  # |score| > 0.15 = moderate signal


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class ArticleSentiment:
    """Sentiment result for a single news article."""
    title: str
    source: str
    published: str
    url: str
    score: float                    # -1 (very negative) to +1 (very positive)
    positive_prob: float
    neutral_prob: float
    negative_prob: float
    label: str                      # "POSITIVE", "NEUTRAL", "NEGATIVE"
    model_used: str                 # "FinBERT" or "VADER"


@dataclass
class SentimentResult:
    """Aggregate sentiment analysis for a ticker."""
    ticker: str
    overall_score: float            # -1 to +1, weighted mean
    signal: str                     # "BULLISH", "NEUTRAL", "BEARISH"
    signal_strength: str            # "STRONG", "MODERATE", "WEAK"
    n_articles: int
    positive_count: int
    neutral_count: int
    negative_count: int
    positive_pct: float
    negative_pct: float
    articles: list[ArticleSentiment]
    model_used: str
    iv_signal: Optional[str]        # "HIGH_VOL_EXPECTED", "LOW_VOL_EXPECTED", None
    timestamp: str


# ── Main engine ───────────────────────────────────────────────────────────────
class SentimentEngine:
    """
    Financial sentiment engine.

    Attempts to load FinBERT from HuggingFace (requires torch + transformers).
    Falls back to VADER if FinBERT is unavailable or if use_vader_fallback=True.

    Parameters
    ----------
    use_vader_fallback : bool
        Force VADER even if FinBERT is available.
    device : str
        'cuda', 'cpu', or 'auto'. Default 'auto'.
    """

    _shared_finbert_pipelines: dict[int, object] = {}
    _shared_vader = None

    def __init__(self, use_vader_fallback: bool = False, device: str = "auto"):
        self.use_vader_fallback = use_vader_fallback
        self._finbert_pipeline = None
        self._vader = None
        self._model_name = "uninitialized"

        # Resolve device
        if device == "auto":
            try:
                import torch
                self._device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                self._device = -1
        elif device == "cuda":
            self._device = 0
        else:
            self._device = -1

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_finbert(self) -> bool:
        """Lazy-load FinBERT pipeline. Returns True on success."""
        if self._finbert_pipeline is not None:
            return True
        shared_pipeline = self.__class__._shared_finbert_pipelines.get(self._device)
        if shared_pipeline is not None:
            self._finbert_pipeline = shared_pipeline
            self._model_name = "FinBERT"
            return True
        try:
            from transformers import pipeline as hf_pipeline
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._finbert_pipeline = hf_pipeline(
                    "text-classification",
                    model=FINBERT_MODEL,
                    tokenizer=FINBERT_MODEL,
                    device=self._device,
                    top_k=None,           # return all 3 class scores
                    truncation=True,
                    max_length=MAX_TOKEN_LENGTH,
                )
            self.__class__._shared_finbert_pipelines[self._device] = self._finbert_pipeline
            self._model_name = "FinBERT"
            return True
        except Exception:
            return False

    def _load_vader(self):
        """Load VADER analyzer."""
        if self._vader is not None:
            return
        shared_vader = self.__class__._shared_vader
        if shared_vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            shared_vader = SentimentIntensityAnalyzer()
            self.__class__._shared_vader = shared_vader
        self._vader = shared_vader
        self._model_name = "VADER"

    def _ensure_model(self):
        """Load the best available model."""
        if self.use_vader_fallback:
            self._load_vader()
            return
        if not self._load_finbert():
            self._load_vader()

    # ── Scoring ───────────────────────────────────────────────────────────────
    def _score_finbert(self, texts: list[str]) -> list[dict]:
        """Score a batch of texts with FinBERT."""
        results = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = self._finbert_pipeline(batch)
            for item in raw:
                label_map = {r["label"].upper(): r["score"] for r in item}
                pos = label_map.get("POSITIVE", 0.0)
                neg = label_map.get("NEGATIVE", 0.0)
                neu = label_map.get("NEUTRAL",  0.0)
                score = pos - neg                   # -1 to +1
                label = max(label_map, key=label_map.get)
                results.append({
                    "score": score, "positive": pos,
                    "neutral": neu, "negative": neg, "label": label
                })
        return results

    def _score_vader(self, texts: list[str]) -> list[dict]:
        """Score texts with VADER."""
        results = []
        for text in texts:
            vs = self._vader.polarity_scores(text)
            score = vs["compound"]          # -1 to +1
            pos   = vs["pos"]
            neg   = vs["neg"]
            neu   = vs["neu"]
            if score >= 0.05:
                label = "POSITIVE"
            elif score <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            results.append({
                "score": score, "positive": pos,
                "neutral": neu, "negative": neg, "label": label
            })
        return results

    def score_texts(self, texts: list[str]) -> list[dict]:
        """
        Score a list of text strings.

        Returns list of dicts with keys:
        score, positive, neutral, negative, label.
        """
        self._ensure_model()
        if self._finbert_pipeline is not None and not self.use_vader_fallback:
            return self._score_finbert(texts)
        return self._score_vader(texts)

    # ── News parsing ──────────────────────────────────────────────────────────
    @staticmethod
    def parse_yfinance_news(news_items: list[dict]) -> list[dict]:
        """
        Normalize yfinance Ticker.news items into a standard format.

        Handles both schema versions:
          Old (yfinance <0.2.37): flat keys — title, publisher, link, providerPublishTime
          New (yfinance >=0.2.37): nested — content.title, content.provider.displayName,
                                            content.canonicalUrl.url, content.pubDate (ISO)
        Returns list of dicts: {title, source, published, url}.
        """
        from datetime import datetime, timezone
        parsed = []
        for item in news_items or []:
            content = item.get("content") or {}
            if content:
                # New nested format
                title  = (content.get("title") or content.get("headline")
                          or item.get("title", "")).strip()
                source = (content.get("provider", {}).get("displayName")
                          or content.get("provider", {}).get("name")
                          or item.get("publisher", "Source unknown")).strip()
                url    = (content.get("canonicalUrl", {}).get("url")
                          or content.get("clickThroughUrl", {}).get("url")
                          or item.get("link", "")).strip()
                pub_date = content.get("pubDate") or content.get("displayTime") or ""
                if pub_date:
                    try:
                        dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                        published = dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    except Exception:
                        published = pub_date[:10]
                else:
                    ts = item.get("providerPublishTime") or 0
                    try:    published = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "Unknown date"
                    except: published = "Unknown date"
            else:
                # Old flat format
                title  = (item.get("title") or item.get("headline") or "").strip()
                source = (item.get("publisher") or item.get("source") or "Source unknown").strip()
                url    = (item.get("link") or item.get("url") or "").strip()
                ts     = item.get("providerPublishTime") or item.get("published") or 0
                try:    published = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts and int(ts) > 0 else "Unknown date"
                except: published = "Unknown date"

            if title:
                parsed.append({"title": title, "source": source,
                                "published": published, "url": url})
        return parsed

    # ── Main analysis ─────────────────────────────────────────────────────────
    def analyze_ticker(self, ticker: str,
                       news_items: list[dict],
                       current_iv: Optional[float] = None) -> SentimentResult:
        """
        Run full sentiment analysis on a ticker's news.

        Parameters
        ----------
        ticker : str
        news_items : list[dict]
            Each dict must have keys: title, source, published, url.
        current_iv : float, optional
            Current implied volatility (decimal). Used to generate IV signal.

        Returns
        -------
        SentimentResult
        """
        from datetime import datetime, timezone

        if not news_items:
            return self._empty_result(ticker)

        texts = [item.get("title", "") for item in news_items]
        texts = [t for t in texts if t.strip()]
        if not texts:
            return self._empty_result(ticker)

        scores = self.score_texts(texts)

        articles = []
        for item, sc in zip(news_items, scores):
            articles.append(ArticleSentiment(
                title=item.get("title", ""),
                source=item.get("source", ""),
                published=item.get("published", ""),
                url=item.get("url", ""),
                score=sc["score"],
                positive_prob=sc["positive"],
                neutral_prob=sc["neutral"],
                negative_prob=sc["negative"],
                label=sc["label"],
                model_used=self._model_name,
            ))

        # Aggregate
        all_scores = np.array([a.score for a in articles])
        # Weight more recent articles higher (first = most recent from yfinance)
        weights = np.exp(-0.15 * np.arange(len(all_scores)))
        weights /= weights.sum()
        overall = float(np.dot(all_scores, weights))

        pos_count = sum(1 for a in articles if a.label == "POSITIVE")
        neg_count = sum(1 for a in articles if a.label == "NEGATIVE")
        neu_count = sum(1 for a in articles if a.label == "NEUTRAL")
        n = len(articles)

        # Signal
        if overall >= STRONG_THRESHOLD:
            signal, strength = "BULLISH", "STRONG"
        elif overall >= MODERATE_THRESHOLD:
            signal, strength = "BULLISH", "MODERATE"
        elif overall <= -STRONG_THRESHOLD:
            signal, strength = "BEARISH", "STRONG"
        elif overall <= -MODERATE_THRESHOLD:
            signal, strength = "BEARISH", "MODERATE"
        else:
            signal, strength = "NEUTRAL", "WEAK"

        # IV signal: negative sentiment + low IV = buy protection
        iv_signal = None
        if current_iv is not None:
            if overall <= -MODERATE_THRESHOLD and current_iv < 0.25:
                iv_signal = "HIGH_VOL_EXPECTED"
            elif overall >= MODERATE_THRESHOLD and current_iv > 0.40:
                iv_signal = "LOW_VOL_EXPECTED"

        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return SentimentResult(
            ticker=ticker,
            overall_score=overall,
            signal=signal,
            signal_strength=strength,
            n_articles=n,
            positive_count=pos_count,
            neutral_count=neu_count,
            negative_count=neg_count,
            positive_pct=pos_count / max(n, 1),
            negative_pct=neg_count / max(n, 1),
            articles=articles,
            model_used=self._model_name,
            iv_signal=iv_signal,
            timestamp=ts,
        )

    def _empty_result(self, ticker: str) -> SentimentResult:
        from datetime import datetime, timezone
        return SentimentResult(
            ticker=ticker, overall_score=0.0, signal="NEUTRAL",
            signal_strength="WEAK", n_articles=0,
            positive_count=0, neutral_count=0, negative_count=0,
            positive_pct=0.0, negative_pct=0.0,
            articles=[], model_used="none", iv_signal=None,
            timestamp=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )

    @property
    def model_name(self) -> str:
        return self._model_name


# ── Standalone helpers ────────────────────────────────────────────────────────
def sentiment_gauge_data(result: SentimentResult) -> dict:
    """
    Returns data for a Plotly indicator gauge chart.
    Score mapped to [-1, +1] range with color zones.
    """
    return {
        "value": result.overall_score,
        "min": -1.0,
        "max": 1.0,
        "signal": result.signal,
        "signal_strength": result.signal_strength,
        "color": "#00D4AA" if result.signal == "BULLISH"
                 else "#F43F5E" if result.signal == "BEARISH"
                 else "#F59E0B",
    }


def articles_dataframe(result: SentimentResult) -> pd.DataFrame:
    """Convert articles list to a styled DataFrame."""
    rows = []
    for a in result.articles:
        rows.append({
            "Title": a.title[:80] + "…" if len(a.title) > 80 else a.title,
            "Source": a.source,
            "Date": a.published[:10],
            "Sentiment": a.label,
            "Score": f"{a.score:+.3f}",
            "Positive": f"{a.positive_prob*100:.1f}%",
            "Negative": f"{a.negative_prob*100:.1f}%",
        })
    return pd.DataFrame(rows)
