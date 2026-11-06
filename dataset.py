"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # Slang / informal / regional additions (breaker test fix)
    "fire",
    "slaps",
    "bussin",
    "hyped",
    "proud",
    "hopeful",
    "chuffed",
    "🔥",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # Slang / informal / regional additions (breaker test fix)
    "mid",
    "gutted",
    "exhausted",
    "shambles",
    "😭",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "lowkey proud of myself for finishing that essay 😌",
    "I absolutely love getting stuck in traffic for an hour 🙃",
    "got the internship but I have to move across the country lol",
    "bro I'm so done with this week 💀",
    "this song has me in my feelings rn 🥲",
    "can't wait to see my friends this weekend!!",
    "had a dentist appointment, now heading to work",
    "not bad for a first try honestly",
    "happy for her, but kinda wish it was me",
    "ngl today was mid",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "positive",  # "lowkey proud of myself for finishing that essay 😌"
    "negative",  # "I absolutely love getting stuck in traffic for an hour 🙃" (sarcasm)
    "mixed",     # "got the internship but I have to move across the country lol"
    "negative",  # "bro I'm so done with this week 💀"
    "mixed",     # "this song has me in my feelings rn 🥲"
    "positive",  # "can't wait to see my friends this weekend!!"
    "neutral",   # "had a dentist appointment, now heading to work"
    "positive",  # "not bad for a first try honestly" (negation)
    "mixed",     # "happy for her, but kinda wish it was me"
    "negative",  # "ngl today was mid" (debatable: some would say neutral)
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
