"""
High-Performance Model Trainer for Knowledge Gap Detection
Predictive Feature Engineering + Tuned XGBoost
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

gap_path = os.path.join(BASE_DIR, "data", "processed", "knowledge_gap_dataset.csv")
clean_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_skill_builder.csv")

print("⏳ Loading Datasets...")
gap = pd.read_csv(gap_path)
clean = pd.read_csv(clean_path)

df = clean.merge(gap[['user_id', 'skill_name', 'KnowledgeGap']], on=['user_id', 'skill_name'], how='inner')
print(f"Merged Dataset Shape: {df.shape}")

# 1. Student Historical Performance Aggregates
user_stats = df.groupby('user_id').agg(
    user_avg_correct=('correct', 'mean'),
    user_avg_attempts=('attempt_count', 'mean'),
    user_avg_hints=('hint_count', 'mean'),
    user_avg_latency=('ms_first_response', 'mean')
).reset_index()

# 2. Skill & User-Skill Mastery Signals
user_skill_stats = df.groupby(['user_id', 'skill_name']).agg(
    user_skill_mastery=('correct', 'mean'),
    user_skill_attempts=('attempt_count', 'mean')
).reset_index()

# 3. Skill Difficulty Baseline Aggregates
skill_stats = df.groupby('skill_name').agg(
    skill_avg_correct=('correct', 'mean'),
    skill_avg_attempts=('attempt_count', 'mean')
).reset_index()

df = df.merge(user_stats, on='user_id', how='left')
df = df.merge(user_skill_stats, on=['user_id', 'skill_name'], how='left')
df = df.merge(skill_stats, on='skill_name', how='left')

# 4. Interaction & High-Gap Feature Engineering
df['hint_ratio'] = df['hint_count'] / (df['hint_total'] + 1)
df['attempt_hint_sum'] = df['attempt_count'] + df['hint_count']
df['log_ms_response'] = np.log1p(df['ms_first_response'].clip(lower=0))
df['log_overlap_time'] = np.log1p(df['overlap_time'].clip(lower=0))
df['is_first_action_hint'] = (df['first_action'] == 1).astype(int)
df['correct_first_try'] = ((df['correct'] == 1) & (df['attempt_count'] == 1) & (df['hint_count'] == 0)).astype(int)
df['high_gap_signal'] = (df['user_skill_mastery'] < 0.6).astype(float)

features = [
    "correct",
    "correct_first_try",
    "user_skill_mastery",
    "high_gap_signal",
    "user_avg_correct",
    "user_avg_attempts",
    "user_avg_hints",
    "user_avg_latency",
    "skill_avg_correct",
    "skill_avg_attempts",
    "attempt_count",
    "hint_count",
    "hint_ratio",
    "attempt_hint_sum",
    "log_ms_response",
    "log_overlap_time",
    "is_first_action_hint",
    "opportunity",
    "position",
    "tutor_mode",
    "answer_type",
    "type",
    "skill_name"
]

X = df[features]
y_raw = df["KnowledgeGap"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

numeric_features = [
    "correct",
    "correct_first_try",
    "user_avg_correct",
    "user_avg_attempts",
    "user_avg_hints",
    "user_avg_latency",
    "skill_avg_correct",
    "skill_avg_attempts",
    "attempt_count",
    "hint_count",
    "hint_ratio",
    "attempt_hint_sum",
    "log_ms_response",
    "log_overlap_time",
    "is_first_action_hint",
    "opportunity",
    "position"
]

categorical_features = [
    "tutor_mode",
    "answer_type",
    "type",
    "skill_name"
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

print("🤖 Training Tuned XGBoost Model...")
xgb_model = XGBClassifier(
    n_estimators=350,
    max_depth=10,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1
)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"🎉 XGBoost Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("="*50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save models
model_save_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
encoder_save_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

joblib.dump(model_pipeline, model_save_path)
joblib.dump(label_encoder, encoder_save_path)

print(f"\n✔ Saved model to {model_save_path}")
print(f"✔ Saved label encoder to {encoder_save_path}")
