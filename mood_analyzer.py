# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

        # Words that flip the meaning of the word right after them.
        self.negation_words = {
            "not", "no", "never", "isn't", "wasn't", "don't",
            "doesn't", "didn't", "ain't", "aren't", "won't",
        }

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          - Strip whitespace and lowercase everything
          - Shorten stretched words ("soooo" -> "soo") so they are consistent
          - Keep text emoticons (":)", ":-(") and emojis as their own tokens
          - Keep apostrophes inside words ("can't", "i'm")
          - Drop all other punctuation ("day!!" -> "day")
        """
        cleaned = text.strip().lower()

        # Any character repeated 3+ times becomes 2 ("sooooo" -> "soo").
        cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)

        # Match, in order: text emoticons, words (with inner apostrophes),
        # or any single non-space, non-word character (emojis, punctuation).
        pattern = r"[:;]-?[)(dp]|[a-z0-9]+(?:'[a-z]+)?|[^\w\s]"
        tokens = re.findall(pattern, cleaned)

        # Throw away plain ASCII punctuation, but keep emojis and emoticons.
        return [t for t in tokens if t.isalnum() or "'" in t or len(t) > 1 or not t.isascii()]

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def _find_hits(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Shared helper: walk the tokens once and return
        (score, positive_hits, negative_hits).

        Enhancement: negation handling. If a sentiment word comes right
        after a negation word ("not happy", "never fun"), its effect flips.
        """
        tokens = self.preprocess(text)
        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []

        for i, token in enumerate(tokens):
            if token in self.positive_words:
                value = 1
            elif token in self.negative_words:
                value = -1
            else:
                continue

            negated = i > 0 and tokens[i - 1] in self.negation_words
            if negated:
                value = -value
                token = f"{tokens[i - 1]} {token}"

            score += value
            if value > 0:
                positive_hits.append(token)
            else:
                negative_hits.append(token)

        return score, positive_hits, negative_hits

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Start at 0. Each positive word adds 1, each negative word subtracts 1.
        A negation right before a word flips it, so "not happy" counts as -1
        and "not bad" counts as +1.
        """
        score, _, _ = self._find_hits(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the matches for a piece of text into a mood label.

          - both positive and negative signals -> "mixed"
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 (no signals) -> "neutral"
        """
        score, positive_hits, negative_hits = self._find_hits(text)

        if positive_hits and negative_hits:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label,
        e.g. 'Score = -1 (positive: [], negative: ['not happy'])'.
        """
        score, positive_hits, negative_hits = self._find_hits(text)
        return f"Score = {score} (positive: {positive_hits}, negative: {negative_hits})"
