import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score, roc_curve, 
                             mean_absolute_error, mean_squared_error, r2_score)

# 1. Generate Dataset & Split (60 / 20 / 20)
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'income': np.random.exponential(50000, n_samples) + 20000,
    'loan_amount': np.random.exponential(15000, n_samples) + 5000,
    'credit_score': np.random.randint(550, 850, n_samples),
    'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Business', 'Unemployed'], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
    'education': np.random.choice(['High School', 'Graduate', 'Post-Graduate'], n_samples, p=[0.3, 0.5, 0.2]),
    'application_date': pd.date_range(start='2023-01-01', periods=n_samples, freq='h'),
    'default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
})

df.loc[df.sample(50, random_state=42).index, 'income'] = np.nan
df.loc[df.sample(30, random_state=42).index, 'employment_type'] = np.nan

X = df.drop(columns=['default'])
y = df['default']

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full)

def engineer_features(data):
    df_f = data.copy()
    df_f['application_date'] = pd.to_datetime(df_f['application_date'])
    df_f['app_month'] = df_f['application_date'].dt.month
    df_f['app_dayofweek'] = df_f['application_date'].dt.dayofweek
    df_f['income_loan_ratio'] = df_f['income'] / (df_f['loan_amount'] + 1)
    df_f['log_income'] = np.log1p(df_f['income'].fillna(df_f['income'].median()))
    df_f = df_f.drop(columns=['application_date'])
    return df_f

numeric_features = ['income', 'loan_amount', 'credit_score', 'app_month', 'app_dayofweek', 'income_loan_ratio', 'log_income']
categorical_features = ['employment_type']
ordinal_features = ['education']
education_categories = [['High School', 'Graduate', 'Post-Graduate']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
        ('ord', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ordinal', OrdinalEncoder(categories=education_categories))]), ordinal_features)
    ]
)

print("=== 1. CLASSIFICATION METRICS (Random Forest) ===")
rf_pipe = Pipeline(steps=[('fe', FunctionTransformer(engineer_features)), ('preprocessor', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
rf_pipe.fit(X_train, y_train)
y_pred = rf_pipe.predict(X_val)

print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print(f"Accuracy: {accuracy_score(y_val, y_pred):.2f}")
print(f"Precision: {precision_score(y_val, y_pred):.2f}")
print(f"Recall: {recall_score(y_val, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_val, y_pred):.2f}")

print("\n=== 2. ROC-AUC SCORE ===")
y_prob = rf_pipe.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_prob)
print(f"ROC-AUC: {roc_auc:.2f}")

print("\n=== 3. HYPERPARAMETER TUNING (GridSearchCV on RF) ===")
param_grid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [3, 5],
    'clf__min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf_pipe, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best RF Parameters:", grid_search.best_params_)
print("Best RF Score:", grid_search.best_score_)

print("\n=== 4. REGRESSION EVALUATION VS BASELINE ===")
reg_features = ['income', 'credit_score']
X_reg = X_train[reg_features].fillna(X_train[reg_features].median())
y_reg = X_train['loan_amount'].fillna(X_train['loan_amount'].median())

lin_reg = LinearRegression().fit(X_reg, y_reg)
y_pred_reg = lin_reg.predict(X_reg)
baseline_pred = np.full_like(y_reg, y_reg.mean())

print(f"MAE: {mean_absolute_error(y_reg, y_pred_reg):.2f} (Baseline: {mean_absolute_error(y_reg, baseline_pred):.2f})")
print(f"RMSE: {np.sqrt(mean_squared_error(y_reg, y_pred_reg)):.2f}")
print(f"R-squared: {r2_score(y_reg, y_pred_reg):.2f}")

print("\n=== 5. FINAL EVALUATION ON TEST SET ===")
final_test_pred = grid_search.best_estimator_.predict(X_test)
final_test_acc = accuracy_score(y_test, final_test_pred)
print(f"Final Best Model Test Accuracy: {final_test_acc:.2f}")