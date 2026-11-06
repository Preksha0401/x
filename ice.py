# =====================================================
# DRIVER CODE - STEP 1: IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# =====================================================
# DRIVER CODE - STEP 2: LOAD TITANIC DATASET
# =====================================================

titanic = sns.load_dataset("titanic")

print(titanic.head())
print(titanic.isnull().sum())


# =====================================================
# MAIN LOGIC - STEP 3: SELECT FEATURES AND TARGET
# =====================================================

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = titanic[features]
y = titanic["survived"]

print(X.head())


# =====================================================
# MAIN LOGIC - STEP 4: IDENTIFY NUMERICAL AND
# CATEGORICAL FEATURES
# =====================================================

numerical_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


# =====================================================
# MAIN LOGIC - STEP 5: PREPROCESSING
# =====================================================

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both transformers
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


# =====================================================
# DRIVER CODE - STEP 6: TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =====================================================
# MAIN LOGIC - STEP 7: TRAIN RANDOM FOREST MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Create complete pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

print("Random Forest trained successfully!")


# =====================================================
# DRIVER CODE - STEP 8: EVALUATE MODEL
# =====================================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =====================================================
# ICE PLOT IMPLEMENTATION
# =====================================================

# =====================================================
# MAIN LOGIC - STEP 9: SELECT FEATURE
# =====================================================

ice_feature = "age"

print("Minimum age:", X_test[ice_feature].min())
print("Maximum age:", X_test[ice_feature].max())


# =====================================================
# MAIN LOGIC - STEP 10: DETERMINE FEATURE RANGE
# =====================================================

lower_bound = X_test[ice_feature].quantile(0.05)
upper_bound = X_test[ice_feature].quantile(0.95)

print("ICE range:", lower_bound, "to", upper_bound)


# =====================================================
# MAIN LOGIC - STEP 11: GENERATE EQUALLY SPACED VALUES
# =====================================================

K = 50

R = np.linspace(
    lower_bound,
    upper_bound,
    K
)


# =====================================================
# MAIN LOGIC - STEP 12: GENERATE INDIVIDUAL ICE
# PREDICTIONS
# =====================================================

ice_predictions = []

for index in X_test.index:

    # Take one passenger
    sample = X_test.loc[[index]].copy()

    sample_predictions = []

    # Change age for this passenger
    for value in R:

        modified_sample = sample.copy()

        modified_sample[ice_feature] = value

        # Predict survival probability for Class 1
        prediction = pipeline.predict_proba(
            modified_sample
        )[0, 1]

        sample_predictions.append(prediction)

    ice_predictions.append(sample_predictions)

ice_predictions = np.array(ice_predictions)

print("ICE prediction shape:", ice_predictions.shape)


# =====================================================
# MAIN LOGIC - STEP 13: PLOT ICE CURVES
# =====================================================

plt.figure(figsize=(10, 6))

for i in range(len(ice_predictions)):

    plt.plot(
        R,
        ice_predictions[i],
        alpha=0.25
    )

# Average ICE curve
average_ice = np.mean(ice_predictions, axis=0)

plt.plot(
    R,
    average_ice,
    color="black",
    linewidth=3,
    label="Average ICE"
)

plt.xlabel("Age")
plt.ylabel("Predicted Probability of Survival")

plt.title("Individual Conditional Expectation (ICE) Plot")

plt.legend()
plt.grid(True)
plt.show()


# =====================================================
# ALE PLOT IMPLEMENTATION
# =====================================================

# =====================================================
# MAIN LOGIC - STEP 14: SELECT FEATURE AND CREATE BINS
# =====================================================

ale_feature = "age"

num_bins = 10

bin_edges = np.linspace(
    X_test[ale_feature].min(),
    X_test[ale_feature].max(),
    num_bins + 1
)

print("Bin edges:", bin_edges)


# =====================================================
# MAIN LOGIC - STEP 15: CALCULATE LOCAL EFFECTS
# =====================================================

local_effects = []
bin_centers = []

for i in range(num_bins):

    lower = bin_edges[i]
    upper = bin_edges[i + 1]

    # Select observations belonging to this bin
    if i == num_bins - 1:
        mask = (
            (X_test[ale_feature] >= lower) &
            (X_test[ale_feature] <= upper)
        )
    else:
        mask = (
            (X_test[ale_feature] >= lower) &
            (X_test[ale_feature] < upper)
        )

    samples = X_test.loc[mask].copy()

    bin_centers.append((lower + upper) / 2)

    if len(samples) == 0:
        local_effects.append(0)
        continue

    # Replace age with lower boundary
    lower_samples = samples.copy()
    lower_samples[ale_feature] = lower

    # Replace age with upper boundary
    upper_samples = samples.copy()
    upper_samples[ale_feature] = upper

    # Predict survival probabilities
    lower_predictions = pipeline.predict_proba(
        lower_samples
    )[:, 1]

    upper_predictions = pipeline.predict_proba(
        upper_samples
    )[:, 1]

    # Calculate difference
    differences = (
        upper_predictions - lower_predictions
    )

    # Average local effect
    average_difference = np.mean(differences)

    local_effects.append(average_difference)


# =====================================================
# MAIN LOGIC - STEP 16: ACCUMULATE LOCAL EFFECTS
# =====================================================

ale_values = np.cumsum(local_effects)

print("Local effects:")
print(local_effects)

print("Accumulated effects:")
print(ale_values)


# =====================================================
# MAIN LOGIC - STEP 17: CENTER ALE VALUES AROUND ZERO
# =====================================================

ale_values = ale_values - np.mean(ale_values)

print("Centered ALE values:")
print(ale_values)


# =====================================================
# MAIN LOGIC - STEP 18: PLOT ALE CURVE
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(
    bin_centers,
    ale_values,
    marker="o"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Age")
plt.ylabel("Accumulated Local Effect")

plt.title("Accumulated Local Effects (ALE) Plot")

plt.grid(True)
plt.show()


# =====================================================
# MAIN LOGIC - STEP 19: DISPLAY ALE RESULTS
# =====================================================

ale_table = pd.DataFrame({
    "Bin Lower": bin_edges[:-1],
    "Bin Upper": bin_edges[1:],
    "Bin Center": bin_centers,
    "Local Effect": local_effects,
    "ALE": ale_values
})

print(ale_table)