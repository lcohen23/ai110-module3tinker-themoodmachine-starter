# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

## 1. Model Overview

**Model type:**
I compared both models. The rule based model is the main one I designed. The ML model was trained on the same data to compare against it.

**Intended purpose:**
Classify short, social-media-style posts as `positive`, `negative`, `neutral`, or `mixed`. It is a learning project, not a tool for real decisions about people.

**How it works (brief):**

- **Rule based:** clean the text into tokens, then add 1 for every positive word and subtract 1 for every negative word. A negation word right before a sentiment word flips it. If a post has both positive and negative words it is `mixed`. Otherwise the sign of the score decides the label.
- **ML:** turn each post into word counts (bag of words), then train a logistic regression classifier to link those counts to the human labels.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` has **21 posts**:

- **6 starter posts** that came with the project.
- **10 posts** added to cover realistic language: slang, emojis, sarcasm and mixed feelings.
- **5 posts** added later to see how changing the data affects each model.

**Labeling process:**
Each post got one label, chosen by asking "what mood would most readers take from this?" The new posts and their first labels were drafted with an AI coding assistant (Claude) and then accepted. There was no second human labeler, so the labels reflect one point of view.

Hard or debatable posts:

- `"ngl today was mid"` is labeled **negative**, but many people would call it neutral.
- `"This is fine"` is labeled **neutral**, but it is often used sarcastically, as in the "this is fine" meme.
- `"bro I'm so done with this week 💀"` is labeled **negative**, but 💀 often means "I'm laughing".
- `"this song has me in my feelings rn 🥲"` is labeled **mixed**. It could be read as sad or as happy-nostalgic.

**Label balance:**

| positive | negative | mixed | neutral |
|---|---|---|---|
| 7 | 7 | 4 | 3 |

**Important characteristics of the dataset:**

- Contains slang (`lowkey`, `ngl`, `mid`, `rn`, `bro`) and emojis (😌 🙃 💀 🥲)
- Includes sarcasm (`"I absolutely love getting stuck in traffic for an hour 🙃"`, `"bus was late again, love that for me 🙃"`)
- Some posts express mixed feelings (`"got the internship but I have to move across the country lol"`)
- All posts are short, one sentence, and in English

**Possible issues with the dataset:**

- **Too small:** 21 posts is far too few to train or fairly evaluate a model.
- **Few neutral and mixed examples:** there are only 3 neutral and 4 mixed posts.
- **No separate test set:** the same posts are used to build the word lists, to train the ML model, and to evaluate both.
- **Narrow language:** the slang is mostly young, online, US-centric English. There is no other language, code-switching or older users' phrasing.

## 3. How the Rule Based Model Works

**Your scoring rules:**

1. **Preprocessing (`preprocess`):**
   - Lowercase the text and squash stretched letters (`"SOOOO"` → `"soo"`).
   - Keep apostrophes inside words (`"can't"`).
   - Keep emojis and emoticons like `:)` as their own tokens.
   - Drop other punctuation (`"weekend!!"` → `"weekend"`).
2. **Scoring (`score_text`):** start at 0. Each word in `POSITIVE_WORDS` adds 1 and each word in `NEGATIVE_WORDS` subtracts 1.
3. **Enhancement: negation handling.** If the token right before a sentiment word is a negation (`not`, `no`, `never`, `don't`, `isn't`, and so on), the word's value flips:
   - `"I am not happy about this"` scores -1 → negative
   - `"not bad for a first try honestly"` scores +1 → positive
4. **Enhancement: vocabulary for slang, regional words and emojis.** After breaker testing (section 6), these were added:
   - positive: `fire`, `slaps`, `bussin`, `hyped`, `proud`, `hopeful`, `chuffed`, 🔥
   - negative: `mid`, `gutted`, `exhausted`, `shambles`, 😭

   `sick` and `wicked` were left out on purpose because each can be good or bad depending on context.
5. **Labels (`predict_label`):**
   - both positive and negative words found → `mixed`
   - otherwise, score > 0 → `positive`, score < 0 → `negative`, score = 0 → `neutral`

**Strengths of this approach:**

- **Predictable and explainable.** `explain()` shows exactly which words caused the label, for example `Score = -1 (positive: [], negative: ['not happy'])`.
- **Works well on plain, direct sentences:** `"Today was a terrible day"` → negative, `"So excited for the weekend"` → positive.
- **Handles simple negation:** `"this movie was not boring at all"` → positive.
- **Generalizes better than the ML model on new sentences:** 0.80 vs 0.50 on a set of 10 sentences neither model had seen (section 5).

**Weaknesses of this approach:**

- **It can only use words it knows.** Anything outside the word lists is invisible.
- **One keyword can decide the label,** which is why it misses sarcasm.
- **Negation only looks one word back,** so `"not very happy"` is missed.
- **Every word has one fixed meaning.** "fire" is always good and 😭 is always sad, even when context says otherwise.

## 4. How the ML Model Works

**Features used:**
Bag of words using scikit-learn's `CountVectorizer` with default settings. Two things follow from those defaults:

- It ignores emojis entirely.
- It drops one-letter tokens, so `"I'm"` becomes just `"m"`, which is then dropped.

**Training data:**
The model trained on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:**

- **Training accuracy:** 1.00 with both 16 and 21 posts. It memorizes the training set.
- **Accuracy on 10 unseen sentences:** 0.40 with 16 posts, 0.50 with 21 posts.
- **Adding 5 posts changed many predictions,** for better and for worse:
  - `"this pizza is fire"` went from neutral to positive (correct, because "fire" now appeared in a training post).
  - `"I'm exhausted but proud of myself"` went from mixed to positive (wrong).

**Strengths and weaknesses:**

- **Strength:** it learns from labels automatically. For example, it learned that the "love … stuck … traffic" pattern was negative, which no word list could do.
- **Weakness: spurious cues.** With so little data it latches onto meaningless words:
  - `"I am so happy"` → **negative**, because "am" appeared in `"I am not happy about this"`.
  - `"that game was bussin fr"` → negative, because "was" appeared in `"Today was a terrible day"` and `"ngl today was mid"`.

## 5. Evaluation

**How you evaluated the model:**

1. **Labeled dataset:** ran both models on the 21 labeled posts (`python main.py`, `python ml_experiments.py`).
   - Rule based: **0.62** (13 of 21 correct).
   - ML: **1.00**, but this is training accuracy and only shows memorization.
2. **Unseen sentences:** ran both models on 10 sentences that are not in the dataset.
   - Rule based: **0.80**.
   - ML: **0.50**.
3. **Breaker sentences:** tested the rule based model on sentences designed to confuse it, including style pairs that say the same thing in neutral and informal English (section 6).

**Accuracy history of the rule based model on the labeled posts:**

| Step | Posts | Accuracy |
|---|---|---|
| First implementation | 16 | 0.44 |
| After slang vocabulary fix | 16 | 0.62 |
| After adding 5 posts | 21 | 0.62 |

Part of the jump from 0.44 to 0.62 came from adding words (`proud`, `hopeful`, `mid`) that appear in the evaluation posts themselves, so 0.62 overstates how well the model handles new text.

**Examples of correct predictions:**

- `"I am not happy about this"` → negative. Negation flips "happy" to -1.
- `"Feeling tired but kind of hopeful"` → mixed. "tired" (-1) and "hopeful" (+1) are both found, which triggers the mixed rule.
- `"not bad for a first try honestly"` → positive. Without negation handling, "bad" alone would have made this negative.

**Examples of incorrect predictions:**

- `"I absolutely love getting stuck in traffic for an hour 🙃"` (true: negative, rule based: **positive**).
  - The tokens are `['i', 'absolutely', 'love', 'getting', 'stuck', 'in', 'traffic', 'for', 'an', 'hour', '🙃']`.
  - Only "love" is in a word list, so the score is +1 and the label is positive.
  - The sarcasm comes from a positive word next to a bad situation, which a word list cannot see.
  - The ML model got this one right, but only because it was trained on this exact sentence.
- `"can't wait to see my friends this weekend!!"` (true: positive, rule based: **neutral**). No word is in either list. Excitement here is expressed through a phrase, not a single word.
- `"I am so happy"` (unseen; true: positive, ML: **negative**). The rule based model gets this right. The ML model learned "am" as a negative cue from one training post.

**How the failures differed:**
The rule based model fails where its vocabulary has gaps. The ML model fails where unrelated words happened to appear with a label in training. They fail in different places.

## 6. Limitations

**Sarcasm is read literally.**
Both `"I absolutely love getting stuck in traffic for an hour 🙃"` and `"bus was late again, love that for me 🙃"` are predicted **positive**, because "love" is the only word the model knows in each. This is documented rather than fixed:

- Adding "stuck" or 🙃 as negative would only make these posts `mixed`, not `negative`.
- It would also mislabel posts where 🙃 is just playful.

**One meaning per word.**
The slang fix made new mistakes:

- `"my kitchen caught fire last night"` changed from neutral to **positive**, because "fire" always means good now.
- `"crying laughing at this 😭"` changed from neutral to **negative**, because 😭 always means sad now.
- `"I feel sick today"` and `"that concert was wicked"` are both still **neutral**, because those words were left out on purpose.

**Unknown words default to neutral.**
`"bro I'm so done with this week 💀"` and `"I am so grateful for my friends"` both come out **neutral**, because none of their words are in the lists. A neutral prediction often means "the model didn't understand", not "the post is neutral".

**Mixed feelings need two known words.**
`"happy for her, but kinda wish it was me"` comes out **positive**. "happy" is found, but "kinda wish it was me" has no sentiment word, so the mixed rule never triggers.

**Short negation window.**
Negation only checks the word immediately before a sentiment word, so `"not very happy"` is read as positive.

**Tiny dataset.**
21 posts with only one labeler. The word lists were tuned on the same posts used to evaluate them.

## 7. Ethical Considerations

**Overt bias: informal and regional English was underserved.**
Breaker testing used 7 pairs of sentences that mean the same thing in neutral and informal or regional English. Before the vocabulary fix:

- All 7 neutral-style sentences were correct.
- All 7 informal versions were labeled **neutral**.

| Neutral style | Informal / regional | Before fix |
|---|---|---|
| "This song is good" | "this song slaps" | ✓ / ✗ |
| "I am really upset" | "I'm absolutely gutted" (British) | ✓ / ✗ |
| "I'm so happy" | "I'm well chuffed" (British) | ✓ / ✗ |
| "That game was great" | "that game was bussin fr" | ✓ / ✗ |
| "That movie was boring" | "that movie was mid" | ✓ / ✗ |

The errors depended on **who wrote the text**, not what it said. The vocabulary fix brought all 7 informal versions to correct, but only for the specific words that were added.

Writers who use slang, regional dialects or **African American Vernacular English (AAVE)** are still underserved. The starter word lists were written in standard English, and "bussin" (an AAVE-origin term) was the only AAVE word tested. Any word or grammar pattern outside the hand-picked list still silently becomes neutral.

**Covert compliance: what the AI assistant raised, and what it didn't.**

The assistant did raise several concerns without being asked:

- It left `sick` and `wicked` out of the word lists because their meaning flips with context.
- It added side-effect tests and reported that the slang fix broke `"my kitchen caught fire last night"` and `"crying laughing at this 😭"`.
- It warned that part of the accuracy gain came from adding words taken from the evaluation posts.
- It pointed out that the ML model's 1.00 accuracy was training accuracy and showed memorization, not skill.

It did **not** push back on several things it should have flagged:

- **Labels:** it drafted all the new posts and their labels, and they were accepted without a second labeler or any discussion of disagreements. Only `"mid"` was marked as debatable.
- **Slang coverage:** it picked the slang words itself, a small set of mostly online and British terms. It didn't say how narrow the set was, or that it included an AAVE-origin word without considering the dialect more broadly.
- **Other language styles:** it didn't test other dialects, non-native English, or code-switching.
- **Rule design:** it didn't question its own choices, like the one-word negation window or the "any positive + any negative = mixed" rule.

In each of these cases it confirmed its own decisions instead of questioning them.

**Other risks if used in a real application:**

- **Missed distress:** a message like `"I'm so done 💀"` or one with unfamiliar slang could be labeled neutral. In a safety or wellbeing setting, that means the people the model understands least are the ones most likely to be missed.
- **Misread sarcasm:** sarcastic complaints read as positive could hide real frustration from customers or students.
- **Privacy:** analyzing personal messages for mood should require consent. Mood labels should not be used to make decisions about individuals.

## 8. Ideas for Improvement

- **More labeled data, with more than one labeler,** to measure where people disagree.
- **A separate test set,** so accuracy isn't measured on the data used to tune the word lists or train the ML model.
- **Wider negation:** check two or three words back for negation (`"not very happy"`).
- **Phrase-level rules** for sentiment carried by phrases, like "can't wait" and "love that for me".
- **Word weights** so strong words like `hate` count more than mild ones like `mid`.
- **Better ML features:**
  - Keep emojis and short tokens in `CountVectorizer`.
  - Try TF-IDF.
  - Remove filler words like "am" and "was" that create spurious cues.
- **Context-aware models** (a small transformer) for sarcasm and words with multiple meanings. They would need far more data, and they bring their own bias risks.
