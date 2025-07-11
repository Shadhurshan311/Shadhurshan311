import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ---------- Set random seeds ----------
np.random.seed(42)
random.seed(42)

# ---------- Load and preprocess data ----------
data = pd.read_csv("C:\\Users\\94772\\Desktop\\FYP\\Samples\\samples_July_7.csv")
data.columns = ['Capacitance', 'Humidity', 'Temperature', 'Thickness', 'Cracktype']
data.columns = data.columns.str.strip()
data = data.drop_duplicates()

label_encoder = LabelEncoder()
data['Cracktype'] = label_encoder.fit_transform(data['Cracktype'])

X = data[['Capacitance', 'Humidity', 'Temperature', 'Thickness']].values
y = data['Cracktype'].values

# ---------- Normalize features ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Grid search hyperparameters ----------
param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'n_estimators': [100, 200]
}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_score = 0
best_params = {}

# ---------- Grid Search with Cross-Validation ----------
for lr in param_grid['learning_rate']:
    for depth in param_grid['max_depth']:
        for est in param_grid['n_estimators']:
            fold_accuracies = []
            print(f"\nTrying: learning_rate={lr}, max_depth={depth}, n_estimators={est}")

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                model = XGBClassifier(
                    learning_rate=lr,
                    max_depth=depth,
                    n_estimators=est,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    random_state=42
                )

                model.fit(X_train_fold, y_train_fold)
                y_val_pred = model.predict(X_val_fold)
                acc = accuracy_score(y_val_fold, y_val_pred)
                fold_accuracies.append(acc)

                print(f"  Fold {fold+1} Accuracy: {acc*100:.2f}%")

            avg_acc = np.mean(fold_accuracies)
            print(f"  → Average Accuracy: {avg_acc*100:.2f}%")

            if avg_acc > best_score:
                best_score = avg_acc
                best_params = {
                    'learning_rate': lr,
                    'max_depth': depth,
                    'n_estimators': est
                }

# ---------- Final Training on Full Data with Best Params ----------
print("\n✅ Best Parameters:")
print(best_params)

# Final train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

final_model = XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

final_model.fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)

# ---------- Evaluation ----------
accuracy = accuracy_score(y_test, y_pred_final)
print(f"\n🔸 Final XGBoost Accuracy: {accuracy * 100:.2f}%\n")
print("🔹 Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=label_encoder.classes_))

# ---------- Confusion Matrix ----------
ConfusionMatrixDisplay.from_estimator(
    final_model,
    X_test,
    y_test,
    display_labels=label_encoder.classes_,
    cmap='Blues',
    values_format='d'
)
plt.title("Final XGBoost Confusion Matrix")
plt.show()
