# -*- coding: utf-8 -*-
"""SAVAnalysis

Workflow for logreg and rf models with visualizations of model steps and accuracy.
Original file is located at
    https://colab.research.google.com/drive/1fCKlDmyHRf2rJKFJ00w2yYjJjyj8VyL-
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree



from google.colab import drive
drive.mount('/content/drive', force_remount=True)


file_path = '/content/drive/MyDrive/SAV_Ultimate_Master_Spreadsheet.xlsx'
savsurvey = pd.read_excel(file_path, sheet_name='NOAA_Survey_Data2')
buoy = pd.read_excel(file_path, sheet_name='BuoyData')

# Ensure Date is parsed
savsurvey['Date'] = pd.to_datetime(savsurvey['Date'], errors='coerce')

def datacleaningandmapping(df):
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df = df[(df['Latitude'] > 38.0) & (df['Latitude'] < 39.0)]
    df = df[(df['Longitude'] > -78.0) & (df['Longitude'] < -76.0)] #Optional, only included this cleaning if coordinates need to be brought back
    return df


savsurvey = datacleaningandmapping(savsurvey)


# 3. Handle categorical encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.set_output(transform="pandas")
encoded_sediment = encoder.fit_transform(savsurvey[['Bottom sediment']]) # Make sure this perfectly matches Excel column name
savsurvey2 = pd.concat([savsurvey.drop(columns=['Bottom sediment']), encoded_sediment], axis=1)


savsurvey2['Erosionmap'] = savsurvey2['Erosion'].map({'P': 1, 'A': 0})
savsurvey2 = savsurvey2.drop(columns=['Erosion'])

# 4.  Shorelines
shore_cols = ['Shoreline 1', 'Shoreline 2', 'Shoreline 3']

savsurvey2['Shoreline_List'] = savsurvey2[shore_cols].apply(
    lambda row: [str(val) for val in row if pd.notna(val) and str(val).strip().title() not in ['None', 'nan', '']],
    axis=1
)


mlb = MultiLabelBinarizer()
encoded_matrix = mlb.fit_transform(savsurvey2['Shoreline_List'])
encoded_df = pd.DataFrame(encoded_matrix, columns=[f"Shore_{c}" for c in mlb.classes_], index=savsurvey2.index)
savsurvey2 = pd.concat([savsurvey2, encoded_df], axis=1)
savsurvey2 = savsurvey2.drop(columns=shore_cols + ['Shoreline_List'])


buoyfeatures = ['Salinity', 'Turb', 'Chl', 'T'] # Strings
buoy['DateTime'] = pd.to_datetime(buoy['DateTime'], errors='coerce')

# Convert 15-minute to Daily
buoy_daily = buoy.set_index('DateTime').resample('D').mean(numeric_only=True)
buoy_lags = pd.DataFrame(index=buoy_daily.index)

# Calculate lags with min_periods to survive missing sensor days
for feature in buoyfeatures:
    if feature in buoy_daily.columns:

        buoy_lags[f'{feature}_mean_15d'] = buoy_daily[feature].rolling(window=15, min_periods=5).mean()
        buoy_lags[f'{feature}_max_15d'] = buoy_daily[feature].rolling(window=15, min_periods=5).max()


# Restore Date column for merging
buoy_lags = buoy_lags.reset_index().rename(columns={'DateTime': 'Date'})


# 6. Merge
ml = pd.merge(savsurvey2, buoy_lags, on="Date", how="left")

# 7. Drop NaNs in critical columns
ml['Water depth'] = pd.to_numeric(ml['Water depth'], errors='coerce')
ml['Secchi depth'] = pd.to_numeric(ml['Secchi depth'], errors='coerce')
ml = ml.dropna()

# 8. Create X and y
# select_dtypes(include=['number']) naturally drops Date, Notes, Surveyor Name, etc.
ml['Day_of_Year'] = ml['Date'].dt.dayofyear
ml = ml.dropna()

X = ml.select_dtypes(include=['number']).drop(columns=['Total SAV']).drop(columns=['Longitude']).drop(columns=['Latitude'])
y = ml['Total SAV']
y_binary = (y > 0).astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_binary,
    test_size=0.2,
    stratify=y_binary,        # Forces balanced target classes in both sets
    random_state=42    # Locks the random seed so your results are reproducible
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(C=0.25, solver='saga', random_state=42)
model.fit(X_train_scaled, y_train)

# --- 1. Logistic Regression (Requires Scaled Data) ---
log_reg = LogisticRegression(
    penalty='l1', C=20, solver='saga', max_iter=500,
    class_weight='balanced', random_state=42
)
log_reg.fit(X_train_scaled, y_train)
log_preds = log_reg.predict(X_test_scaled)

print("=== Logistic Regression (No Spatial) ===")
print(f"Accuracy: {accuracy_score(y_test, log_preds):.2f}")
print(classification_report(y_test, log_preds))

# --- 2. Random Forest (Uses Raw Unscaled Data) ---
rf = RandomForestClassifier(
    n_estimators=500, max_depth=5, min_samples_leaf=3,
    class_weight='balanced', random_state=42
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("=== Random Forest (No Spatial) ===")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.2f}")
print(classification_report(y_test, rf_preds))

# --- Permutation Importance on Test Set ---
perm_result = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42
)
perm_importances = pd.Series(perm_result.importances_mean, index=X.columns)
top_perm = perm_importances.sort_values(ascending=False).head(10)

print("\nTop 10 Random Forest Features (Permutation):")
print(top_perm)



# --- Logistic Regression Feature Importance (Coefficients) ---

log_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0],
    'Absolute_Weight': np.abs(log_reg.coef_[0])
})

# Sort by the strongest absolute impact
top_log = log_importance.sort_values(by='Absolute_Weight', ascending=False)

print("\nTop 10 Logistic Regression Features (Absolute Coefficients):")
print(top_log.head(10).to_string(index=False))

# Identify which features the L1 penalty completely removed
zeroed_features = top_log[top_log['Absolute_Weight'] == 0]['Feature'].tolist()
print(f"\nFeatures completely eliminated by L1 penalty ({len(zeroed_features)}):")
print(zeroed_features)

explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values[:, :, 1])

# Local waterfall plot for the first test sample's prediction of Class 1
shap.plots.waterfall(shap_values[0, :, 1])


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

# Initialize the model once outside the loop
rf = RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_leaf=3,
    class_weight='balanced', random_state=42
)

# Loop through the 5 different data chunks
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_binary)):

    # 1. Grab the specific rows for this fold using .iloc
    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_train_fold = y_binary.iloc[train_idx]
    y_test_fold = y_binary.iloc[test_idx]

    # 2. Train and predict on this specific chunk
    rf.fit(X_train_fold, y_train_fold)
    rf_preds = rf.predict(X_test_fold)

    # 3. Grade the fold
    score = accuracy_score(y_test_fold, rf_preds)
    scores.append(score)
    print(f'Fold {fold+1} Accuracy: {score:.4f}')

print(f'\nMean Accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}')

sub_tree = rf.estimators_[0]

# 4. Plot the extracted tree
plt.figure(figsize=(15, 10))  # Resize the figure so the text is clear
plot_tree(
    sub_tree,
    feature_names=X.columns.tolist(),  # Injects your actual column names!
    class_names=['SAV Absent', 'SAV Present'],
    filled=True,  # Colors nodes by majority class/purity
    rounded=True,  # Rounds node box corners
    fontsize=10,
)
plt.title("Visualizing Tree #0 from Random Forest")
plt.show()

# 1. Regenerate fresh predictions for the global 29-row test set
rf_preds_global = rf.predict(X_test)

# 2. Grab the original rows for your test set (so we get the GPS coordinates back)
results_df = ml.loc[X_test.index, ['Latitude', 'Longitude', 'Total SAV', 'Water depth', 'Bottom sediment']].copy()

# 3. Add the true binary labels and the model's predictions
results_df['Actual_SAV_Binary'] = y_test
results_df['Predicted_SAV'] = rf_preds_global

# 4. Filter for Restoration Targets (Actual = 0, Predicted = 1)
restoration_sites = results_df[(results_df['Actual_SAV_Binary'] == 0) & (results_df['Predicted_SAV'] == 1)]

print(f"Found {len(restoration_sites)} Prime Restoration Targets:")
display(restoration_sites)
