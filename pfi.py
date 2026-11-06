# ==================================================
# DRIVER CODE – IMPORT LIBRARIES
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ==================================================
# DRIVER CODE – LOAD DATASET
# ==================================================

df = pd.read_csv("diabetes.csv")

# Separate features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================================
# MAIN LOGIC 1 – TRAIN DECISION TREE
# ==================================================

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)


# ==================================================
# MAIN LOGIC 2 – MAKE PREDICTIONS
# ==================================================

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==================================================
# MAIN LOGIC 3 – MANUAL PERMUTATION FEATURE IMPORTANCE
# ==================================================

# Calculate original accuracy
original_accuracy = accuracy_score(
    y_test, model.predict(X_test)
)

importance = {}

# Repeat for every feature
for feature in X_test.columns:

    # DRIVER: Copy test data
    X_permuted = X_test.copy()

    # MAIN: Shuffle one feature
    X_permuted[feature] = np.random.permutation(
        X_permuted[feature]
    )

    # MAIN: Predict using shuffled data
    y_permuted = model.predict(X_permuted)

    # MAIN: Calculate new accuracy
    new_accuracy = accuracy_score(
        y_test, y_permuted
    )

    # MAIN: Calculate feature importance
    importance[feature] = (
        original_accuracy - new_accuracy
    )


# ==================================================
# MAIN LOGIC 4 – DISPLAY FEATURE IMPORTANCE
# ==================================================

print("\nFeature Importance:")

for feature, score in importance.items():
    print(feature, ":", score)


# ==================================================
# DRIVER CODE – PLOT RESULTS
# ==================================================

plt.bar(importance.keys(), importance.values())

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Permutation Feature Importance")

plt.xticks(rotation=90)

plt.show()