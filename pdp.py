# =====================================================
# DRIVER CODE - STEP 1: IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


# =====================================================
# DRIVER CODE - STEP 2: LOAD DATASET
# =====================================================

df = pd.read_csv("housing.csv")

print(df.head())
print(df.isnull().sum())


# =====================================================
# MAIN LOGIC - STEP 3: IDENTIFY NUMERICAL AND
# CATEGORICAL COLUMNS
# =====================================================

numerical_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_cols = df.select_dtypes(
    include=["object"]
).columns

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)


# =====================================================
# MAIN LOGIC - STEP 4: CONVERT NUMERICAL COLUMNS
# =====================================================

for col in numerical_cols:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )


# =====================================================
# MAIN LOGIC - STEP 5: ENCODE CATEGORICAL FEATURES
# =====================================================

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

# Fill missing values
df = df.fillna(df.median(numeric_only=True))


# =====================================================
# DRIVER CODE - STEP 6: SEPARATE FEATURES AND TARGET
# =====================================================

# Change target name if your CSV uses a different name
target = "sale_price"

X = df.drop(target, axis=1)
y = df[target]

print("Features:", X.shape)
print("Target:", y.shape)


# =====================================================
# DRIVER CODE - STEP 7: TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# =====================================================
# MAIN LOGIC - STEP 8: TRAIN RANDOM FOREST REGRESSOR
# =====================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("Random Forest trained successfully!")


# =====================================================
# DRIVER CODE - STEP 9: MAKE PREDICTIONS
# =====================================================

y_pred = rf_model.predict(X_test)

print("Actual prices:")
print(y_test.head())

print("Predicted prices:")
print(y_pred[:5])


# =====================================================
# DRIVER CODE - STEP 10: EVALUATION METRICS
# =====================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# =====================================================
# MAIN LOGIC - STEP 11: SELECT THREE FEATURES
# =====================================================

pdp_features = [
    "carpet_area",
    "Estimated Value",
    "distance_to_school"
]

print("Features selected for PDP:")
print(pdp_features)


# =====================================================
# MAIN LOGIC - STEP 12: GENERATE EQUALLY SPACED VALUES
# =====================================================

for feature in pdp_features:

    min_value = X_test[feature].min()
    max_value = X_test[feature].max()

    # Generate 50 equally spaced values
    feature_values = np.linspace(
        min_value,
        max_value,
        50
    )

    print(feature)
    print(feature_values)


# =====================================================
# MAIN LOGIC - STEP 13: CALCULATE PDP MANUALLY
# =====================================================

def calculate_pdp(model, X_test, feature, K=50):

    # Step 1: Find minimum and maximum values
    min_value = X_test[feature].min()
    max_value = X_test[feature].max()

    # Step 2: Generate equally spaced values
    feature_values = np.linspace(
        min_value,
        max_value,
        K
    )

    # Step 3: Store average predictions
    pdp = []

    # Step 4: Repeat for each feature value
    for value in feature_values:

        # Make a copy of test data
        X_temp = X_test.copy()

        # Replace selected feature with current value
        X_temp[feature] = value

        # Predict using trained model
        predictions = model.predict(X_temp)

        # Calculate average prediction
        average_prediction = np.mean(predictions)

        # Store result
        pdp.append(average_prediction)

    return feature_values, np.array(pdp)


# =====================================================
# MAIN LOGIC - STEP 14: CALCULATE PDP FOR ALL FEATURES
# =====================================================

pdp_results = {}

for feature in pdp_features:

    values, pdp = calculate_pdp(
        rf_model,
        X_test,
        feature,
        K=50
    )

    pdp_results[feature] = {
        "values": values,
        "PDP": pdp
    }

print("PDP calculation completed!")


# =====================================================
# MAIN LOGIC - STEP 15: PLOT PDP
# =====================================================

for feature in pdp_features:

    values = pdp_results[feature]["values"]
    pdp = pdp_results[feature]["PDP"]

    plt.figure(figsize=(8, 5))

    plt.plot(values, pdp, linewidth=2)

    plt.xlabel(feature)
    plt.ylabel("Average Predicted Sale Price")

    plt.title("Partial Dependence Plot - " + feature)

    plt.grid(True)
    plt.show()


# =====================================================
# MAIN LOGIC - STEP 16: SENSITIVITY ANALYSIS FUNCTION
# =====================================================

def sensitivity_analysis(model, X_test, feature):

    # Calculate original average feature value
    base_value = X_test[feature].mean()

    # Percentage changes
    percentage_changes = [-20, -10, 0, 10, 20]

    results = []

    for change in percentage_changes:

        # Calculate changed feature value
        changed_value = base_value * (1 + change / 100)

        # Copy test data
        X_temp = X_test.copy()

        # Replace feature column
        X_temp[feature] = changed_value

        # Predict house prices
        predictions = model.predict(X_temp)

        # Calculate average predicted price
        avg_price = np.mean(predictions)

        results.append({
            "Change (%)": change,
            "Feature Value": changed_value,
            "Average Predicted Price": avg_price
        })

    return pd.DataFrame(results)


# =====================================================
# MAIN LOGIC - STEP 17: PERFORM SENSITIVITY ANALYSIS
# =====================================================

sensitivity_results = {}

for feature in pdp_features:

    result = sensitivity_analysis(
        rf_model,
        X_test,
        feature
    )

    sensitivity_results[feature] = result

    print("\nSensitivity Analysis:", feature)
    print(result)


# =====================================================
# MAIN LOGIC - STEP 18: PLOT SENSITIVITY ANALYSIS
# =====================================================

for feature in pdp_features:

    result = sensitivity_results[feature]

    plt.figure(figsize=(7, 5))

    plt.plot(
        result["Change (%)"],
        result["Average Predicted Price"],
        marker="o"
    )

    plt.xlabel("Percentage Change (%)")
    plt.ylabel("Average Predicted Sale Price")

    plt.title("Sensitivity Analysis - " + feature)

    plt.grid(True)
    plt.show()