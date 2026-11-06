# ==========================================
# DRIVER CODE – IMPORT LIBRARIES
# ==========================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer


# ==========================================
# DRIVER CODE – LOAD DISTILBERT MODEL
# ==========================================

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name
)

model.eval()

# Class 0 = NEGATIVE, Class 1 = POSITIVE
class_names = ["NEGATIVE", "POSITIVE"]

print("Label mapping:", class_names)


# ==========================================
# MAIN LOGIC – PREDICTION FUNCTION
# ==========================================

# Store sentences evaluated by LIME
perturbation_log = []

def predict_proba(sentences):

    # Record the sentences sent to the model
    perturbation_log.extend(sentences)

    # Tokenize sentences
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    # Model prediction
    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    # Return both class probabilities
    return probabilities.numpy()


# ==========================================
# DRIVER CODE – TEST ORIGINAL SENTENCE
# ==========================================

sentence = "The plot was weak, but the acting was wonderful."

prediction = predict_proba([sentence])[0]

print("\nOriginal Sentence:", sentence)
print("Negative Probability:", prediction[0])
print("Positive Probability:", prediction[1])

print("Predicted Sentiment:", class_names[np.argmax(prediction)])


# ==========================================
# DRIVER CODE – CREATE LIME EXPLAINER
# ==========================================

explainer = LimeTextExplainer(
    class_names=class_names
)


# ==========================================
# MAIN LOGIC – GENERATE LIME EXPLANATION
# ==========================================

perturbation_log = []

explanation = explainer.explain_instance(
    sentence,
    predict_proba,
    num_features=10,
    num_samples=1000
)


# ==========================================
# DRIVER CODE – DISPLAY WORD IMPORTANCE
# ==========================================

word_scores = explanation.as_list(label=1)

print("\nLIME Word Importance:")

for word, score in word_scores:
    print(word, ":", score)


# ==========================================
# DRIVER CODE – PLOT LIME EXPLANATION
# ==========================================

words = [item[0] for item in word_scores]
scores = [item[1] for item in word_scores]

plt.figure(figsize=(9, 5))

plt.barh(words, scores)
plt.axvline(0, color="black")

plt.xlabel("LIME Score")
plt.ylabel("Words")
plt.title("LIME Text Explanation")

plt.show()


# ==========================================
# DRIVER CODE – DISPLAY PERTURBED SENTENCES
# ==========================================

# Remove duplicate sentences while preserving order
unique_sentences = list(dict.fromkeys(perturbation_log))

# Get model probabilities for recorded sentences
perturbation_predictions = predict_proba(unique_sentences)

# Sort by positive probability
sorted_data = sorted(
    zip(unique_sentences, perturbation_predictions),
    key=lambda x: x[1][1]
)

print("\nOriginal Sentence:")
print(sentence)

print("\n8 Random Perturbed Sentences:")

random_indices = np.random.choice(
    len(unique_sentences),
    min(8, len(unique_sentences)),
    replace=False
)

for index in random_indices:
    text, probability = unique_sentences[index], perturbation_predictions[index]

    print(text)
    print("Positive Probability:", probability[1])


print("\n3 Lowest Positive Probability Sentences:")

for text, probability in sorted_data[:3]:
    print(text)
    print("Positive Probability:", probability[1])


print("\n3 Highest Positive Probability Sentences:")

for text, probability in sorted_data[-3:]:
    print(text)
    print("Positive Probability:", probability[1])


# ==========================================
# MAIN LOGIC – REMOVE ONE WORD AT A TIME
# ==========================================

words_in_sentence = sentence.split()

original_probability = predict_proba([sentence])[0][1]

print("\nRemove-One-Word Experiment")

for word in words_in_sentence:

    # Remove one occurrence of the word
    modified_sentence = sentence.replace(word, "", 1)

    # Predict modified sentence
    new_probability = predict_proba([modified_sentence])[0][1]

    # Calculate probability change
    change = original_probability - new_probability

    print("\nRemoved Word:", word)
    print("Modified Sentence:", modified_sentence)
    print("New Positive Probability:", new_probability)
    print("Probability Drop:", change)


# ==========================================
# DRIVER CODE – COMPARE 1000 AND 100 SAMPLES
# ==========================================

# Explanation using 1000 samples
explanation_1000 = explainer.explain_instance(
    sentence,
    predict_proba,
    num_features=10,
    num_samples=1000
)

# Explanation using 100 samples
explanation_100 = explainer.explain_instance(
    sentence,
    predict_proba,
    num_features=10,
    num_samples=100
)


# ==========================================
# DRIVER CODE – DISPLAY COMPARISON
# ==========================================

scores_1000 = dict(explanation_1000.as_list(label=1))
scores_100 = dict(explanation_100.as_list(label=1))

print("\nComparison of LIME Scores")

print("Word\t\t1000 Samples\t100 Samples")

for word in set(scores_1000) | set(scores_100):

    print(
        word,
        "\t",
        round(scores_1000.get(word, 0), 6),
        "\t",
        round(scores_100.get(word, 0), 6)
    )