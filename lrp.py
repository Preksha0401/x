# ==========================================
# DRIVER CODE – IMPORT LIBRARIES
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================
# DRIVER CODE – CREATE SENTIMENT DATASET
# ==========================================

sentences = [
    "i love this movie",
    "this movie is wonderful",
    "the film is amazing",
    "i really enjoyed this movie",
    "excellent acting and story",
    "the movie was fantastic",
    "i like this film",
    "this is a great movie",
    "the story is beautiful",
    "absolutely wonderful experience",
    "the film was enjoyable",
    "i loved the acting",

    "i hate this movie",
    "this movie is terrible",
    "the film is boring",
    "i disliked this movie",
    "awful acting and story",
    "the movie was horrible",
    "i do not like this film",
    "this is a bad movie",
    "the story is disappointing",
    "absolutely terrible experience",
    "the film was unpleasant",
    "i hated the acting"
]

labels = np.array(
    [1] * 12 + [0] * 12
)

# 1 = Positive, 0 = Negative


# ==========================================
# DRIVER CODE – CONVERT TEXT INTO FEATURES
# ==========================================

vectorizer = CountVectorizer(binary=True)

X = vectorizer.fit_transform(sentences).toarray()

y = labels

feature_names = vectorizer.get_feature_names_out()

print("Dataset shape:", X.shape)
print("Vocabulary:", feature_names)
print("Vocabulary size:", len(feature_names))


# ==========================================
# DRIVER CODE – SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=8,
    random_state=42,
    stratify=y
)


# ==========================================
# DRIVER CODE – INITIALIZE NEURAL NETWORK
# ==========================================

np.random.seed(42)

input_size = X.shape[1]
hidden_size = 8

W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros(hidden_size)

W2 = np.random.randn(hidden_size, 1) * 0.1
b2 = np.zeros(1)


# ==========================================
# MAIN LOGIC – ACTIVATION FUNCTIONS
# ==========================================

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


# ==========================================
# MAIN LOGIC – FORWARD PASS
# ==========================================

def forward(X, W1, b1, W2, b2):

    z1 = np.dot(X, W1) + b1

    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2

    probability = sigmoid(z2)

    return z1, a1, z2, probability


# ==========================================
# MAIN LOGIC – TRAIN NEURAL NETWORK
# ==========================================

learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):

    z1, a1, z2, probability = forward(
        X_train, W1, b1, W2, b2
    )

    # Calculate output error
    error = probability - y_train.reshape(-1, 1)

    # Backpropagation
    dz2 = error

    dW2 = np.dot(a1.T, dz2) / len(X_train)
    db2 = np.mean(dz2, axis=0)

    da1 = np.dot(dz2, W2.T)

    dz1 = da1 * (z1 > 0)

    dW1 = np.dot(X_train.T, dz1) / len(X_train)
    db1 = np.mean(dz1, axis=0)

    # Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


# ==========================================
# DRIVER CODE – TEST MODEL
# ==========================================

_, _, _, test_prob = forward(
    X_test, W1, b1, W2, b2
)

predictions = (test_prob.flatten() >= 0.5).astype(int)

print("Training accuracy:",
      accuracy_score(y_train,
                     forward(X_train, W1, b1, W2, b2)[3].flatten() >= 0.5))

print("Testing accuracy:",
      accuracy_score(y_test, predictions))


# ==========================================
# DRIVER CODE – SELECT INPUT SENTENCE
# ==========================================

test_sentence = "i love this movie"

input_vector = vectorizer.transform(
    [test_sentence]
).toarray()[0]


# ==========================================
# MAIN LOGIC – LRP-0 IMPLEMENTATION
# ==========================================

def calculate_lrp(x, W1, b1, W2, b2, epsilon=0):

    # Forward pass for one input
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    output = sigmoid(z2)[0]

    # Initialize output relevance with output score
    R_output = output

    # Output-to-hidden relevance
    contribution_output = a1 * W2.flatten()

    denominator_output = np.sum(contribution_output) + epsilon

    if abs(denominator_output) < 1e-10:
        R_hidden = np.zeros_like(a1)
    else:
        R_hidden = (
            contribution_output / denominator_output
        ) * R_output

    # Hidden-to-input relevance
    R_input = np.zeros_like(x, dtype=float)

    for k in range(len(a1)):

        contribution_input = x * W1[:, k]

        denominator_input = (
            np.sum(contribution_input) + epsilon
        )

        if abs(denominator_input) > 1e-10:

            R_input += (
                contribution_input / denominator_input
            ) * R_hidden[k]

    return R_input


# ==========================================
# DRIVER CODE – CALCULATE LRP SCORES
# ==========================================

lrp_scores = calculate_lrp(
    input_vector,
    W1, b1, W2, b2,
    epsilon=0
)

print("\nLRP Word Relevance:")

for word, score in zip(feature_names, lrp_scores):

    if input_vector[list(feature_names).index(word)] == 1:
        print(word, ":", score)


# ==========================================
# MAIN LOGIC – GRADIENT × INPUT
# ==========================================

def gradient_input(x, W1, b1, W2):

    # Forward pass
    z1 = np.dot(x, W1) + b1

    # ReLU derivative
    relu_mask = (z1 > 0).astype(float)

    # Gradient of output logit with respect to input
    gradient = np.dot(
        W1,
        relu_mask * W2.flatten()
    )

    # Multiply gradient by input
    scores = x * gradient

    return scores


# ==========================================
# DRIVER CODE – CALCULATE GRADIENT × INPUT
# ==========================================

gradient_scores = gradient_input(
    input_vector,
    W1, b1, W2
)

print("\nGradient × Input Scores:")

for word, score in zip(feature_names, gradient_scores):

    if input_vector[list(feature_names).index(word)] == 1:
        print(word, ":", score)


# ==========================================
# DRIVER CODE – COMPARE ATTRIBUTIONS
# ==========================================

present_words = np.where(input_vector == 1)[0]

words = feature_names[present_words]

plt.figure(figsize=(9, 5))

x_pos = np.arange(len(words))

plt.bar(
    x_pos - 0.2,
    lrp_scores[present_words],
    width=0.4,
    label="LRP-0"
)

plt.bar(
    x_pos + 0.2,
    gradient_scores[present_words],
    width=0.4,
    label="Gradient x Input"
)

plt.xticks(x_pos, words)
plt.ylabel("Attribution Score")
plt.title("LRP vs Gradient x Input")

plt.legend()
plt.show()


# ==========================================
# MAIN LOGIC – BIAS-FREE MODEL
# ==========================================

# Set biases to zero while keeping learned weights unchanged

b1_zero = np.zeros_like(b1)
b2_zero = np.zeros_like(b2)

lrp_no_bias = calculate_lrp(
    input_vector,
    W1, b1_zero, W2, b2_zero,
    epsilon=0
)

gradient_no_bias = gradient_input(
    input_vector,
    W1, b1_zero, W2
)

print("\nBias-Free LRP:", lrp_no_bias)
print("Bias-Free Gradient x Input:", gradient_no_bias)


# ==========================================
# MAIN LOGIC – TEST DIFFERENT EPSILON VALUES
# ==========================================

epsilon_values = [0, 0.1, 0.5, 1, 2]

for eps in epsilon_values:

    scores = calculate_lrp(
        input_vector,
        W1, b1, W2, b2,
        epsilon=eps
    )

    print("\nEpsilon:", eps)

    for word, score in zip(feature_names, scores):

        if word in words:
            print(word, ":", score)


# ==========================================
# DRIVER CODE – PLOT LRP RELEVANCE
# ==========================================

plt.figure(figsize=(8, 4))

plt.bar(words, lrp_scores[present_words])

plt.xlabel("Words")
plt.ylabel("Relevance")
plt.title("LRP Word-Level Relevance")

plt.show()