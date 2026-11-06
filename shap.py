# =====================================================
# DRIVER CODE - STEP 1: IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# =====================================================
# DRIVER CODE - STEP 2: LOAD TITANIC DATASET
# =====================================================

df = sns.load_dataset("titanic")

print(df.head())
print(df.isnull().sum())


# =====================================================
# MAIN LOGIC - STEP 3: DATA PREPROCESSING
# =====================================================

# Select required columns
df = df[[
    "survived", "pclass", "sex", "age",
    "sibsp", "parch", "fare", "embarked"
]]

# Handle missing values
df["age"] = df["age"].fillna(df["age"].median())
df["fare"] = df["fare"].fillna(df["fare"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Convert categorical columns into numerical columns
df = pd.get_dummies(
    df,
    columns=["sex", "embarked"],
    drop_first=True,
    dtype=int
)

# Separate features and target
X = df.drop("survived", axis=1)
y = df["survived"]

print("Features:", X.columns.tolist())


# =====================================================
# DRIVER CODE - STEP 4: SPLIT DATASET
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# MAIN LOGIC - STEP 5: TRAIN RANDOM FOREST
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# =====================================================
# DRIVER CODE - STEP 6: MODEL EVALUATION
# =====================================================

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =====================================================
# MAIN LOGIC - STEP 7: RANDOM FOREST FEATURE IMPORTANCE
# =====================================================

importance = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(X.columns, importance)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()


# =====================================================
# MAIN LOGIC - STEP 8: CREATE SHAP TREE EXPLAINER
# =====================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

print("SHAP values calculated")


# =====================================================
# MAIN LOGIC - STEP 9: SELECT SHAP VALUES FOR EACH CLASS
# =====================================================

# Handle different SHAP library output formats
if isinstance(shap_values, list):
    shap_class_0 = shap_values[0]
    shap_class_1 = shap_values[1]

elif len(shap_values.shape) == 3:
    shap_class_0 = shap_values[:, :, 0]
    shap_class_1 = shap_values[:, :, 1]

else:
    shap_class_0 = -shap_values
    shap_class_1 = shap_values


# =====================================================
# MAIN LOGIC - STEP 10: SHAP SUMMARY PLOT - SURVIVED
# =====================================================

shap.summary_plot(
    shap_class_1,
    X_test,
    show=True
)


# =====================================================
# MAIN LOGIC - STEP 11: SHAP SUMMARY PLOT - NOT SURVIVED
# =====================================================

shap.summary_plot(
    shap_class_0,
    X_test,
    show=True
)


# =====================================================
# MAIN LOGIC - STEP 12: SHAP FEATURE IMPORTANCE BAR PLOT
# =====================================================

shap.summary_plot(
    shap_class_1,
    X_test,
    plot_type="bar",
    show=True
)


# =====================================================
# MAIN LOGIC - STEP 13: LOCAL EXPLANATION - WATERFALL
# =====================================================

# Select one test observation
i = 0

# Get base value for class 1
base_value = explainer.expected_value

if isinstance(base_value, (list, np.ndarray)):
    base_value = base_value[1]

# Create SHAP explanation object
explanation = shap.Explanation(
    values=shap_class_1[i],
    base_values=base_value,
    data=X_test.iloc[i].values,
    feature_names=X_test.columns
)

shap.plots.waterfall(explanation)


# =====================================================
# MAIN LOGIC - STEP 14: SHAP FORCE PLOT
# =====================================================

shap.force_plot(
    base_value,
    shap_class_1[i],
    X_test.iloc[i],
    matplotlib=True
)

plt.show()


# =====================================================
# MAIN LOGIC - STEP 15: SHAP DEPENDENCE PLOTS
# =====================================================

shap.dependence_plot(
    "age",
    shap_class_1,
    X_test
)

shap.dependence_plot(
    "fare",
    shap_class_1,
    X_test
)

shap.dependence_plot(
    "pclass",
    shap_class_1,
    X_test
)