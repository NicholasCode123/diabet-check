# train_and_save.py
# Jalankan script ini SEKALI untuk melatih model dan menyimpannya ke file .pkl
# Command: python train_and_save.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix


df = pd.read_csv('diabetes.csv')

x_df = df.drop('Outcome', axis=1)
y_df = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x_df, y_df, test_size=0.3, random_state=42, stratify=y_df
)

# Handle zeros (data leakage-safe: fit only on train)
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x_train[cols_with_zeros] = x_train[cols_with_zeros].replace(0, np.nan)
train_medians = x_train[cols_with_zeros].median()
x_train[cols_with_zeros] = x_train[cols_with_zeros].fillna(train_medians)
x_test[cols_with_zeros] = x_test[cols_with_zeros].replace(0, np.nan)
x_test[cols_with_zeros] = x_test[cols_with_zeros].fillna(train_medians)

# Scale
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Training models with GridSearchCV (this may take a few minutes)...")

param_grid_lg = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}
grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    param_grid_lg, cv=5, scoring="f1", n_jobs=-1
)



param_grid_rf = {
    "n_estimators": [200, 300, 400],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid_rf, cv=5, scoring="f1", n_jobs=-1
)

grid_lr.fit(x_train_scaled, y_train)
print(f"  LR best params: {grid_lr.best_params_}")
# grid_xgb.fit(x_train_scaled, y_train)
# print(f"  XGB best params: {grid_xgb.best_params_}")
grid_rf.fit(x_train_scaled, y_train)
print(f"  RF best params: {grid_rf.best_params_}")

ensemble_model = VotingClassifier(
    estimators=[
        ('lr', grid_lr.best_estimator_),
        ('rf', grid_rf.best_estimator_)
    ],
    voting='soft'
)
ensemble_model.fit(x_train_scaled, y_train)
y_prob_test = ensemble_model.predict_proba(x_test_scaled)[:, 1]
y_pred_test = (y_prob_test >= 0.5).astype(int)

print("=== TEST FINAL ===")
print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))


model_package = {
    'model': ensemble_model,
    'scaler': scaler,
    'train_medians': train_medians,
    'cols_with_zeros': cols_with_zeros,
    'threshold': 0.35,
    'feature_names': list(x_df.columns)
}

with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("\n✅ Model saved to diabetes_model.pkl")
print("Now run: python app.py")
