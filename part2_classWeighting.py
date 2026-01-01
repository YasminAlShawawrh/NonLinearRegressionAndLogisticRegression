import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns   

# Load dataset 
csv_file = "C:/Users/Asus/OneDrive/Documents/ML_Ass2/customer_data.csv"
TARGET_COLUMN = "ChurnStatus"

data = pd.read_csv(csv_file)

# Fix missing values
data = data.fillna(data.mean(numeric_only=True))

# Target
y_raw = data[TARGET_COLUMN]

# Convert Yes/No → 0/1 if needed
if y_raw.dtype == "O":
    y = (y_raw == "Yes").astype(int)
else:
    y = y_raw

# Features: drop target + ID
X = data.drop(columns=[TARGET_COLUMN, "CustomerID"])

# One-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

print("X shape:", X.shape)
print("Total rows:", len(X))

# Train/val/test split
total_samples = len(X)

if total_samples < 3500:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp
    )
else:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=2500, random_state=0, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp
    )

print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Metric helper 
def evaluate_model(model, Xtr, ytr, Xv, yv, Xte, yte):
    metrics = {}
    
    ytr_pred = model.predict(Xtr)
    yv_pred  = model.predict(Xv)
    yte_pred = model.predict(Xte)

    yv_proba  = model.predict_proba(Xv)[:, 1]
    yte_proba = model.predict_proba(Xte)[:, 1]

    metrics["train_acc"] = accuracy_score(ytr, ytr_pred)
    metrics["train_prec"] = precision_score(ytr, ytr_pred, zero_division=0)
    metrics["train_rec"]  = recall_score(ytr, ytr_pred, zero_division=0)

    metrics["val_acc"] = accuracy_score(yv, yv_pred)
    metrics["val_prec"] = precision_score(yv, yv_pred, zero_division=0)
    metrics["val_rec"] = recall_score(yv, yv_pred, zero_division=0)

    metrics["test_acc"] = accuracy_score(yte, yte_pred)
    metrics["test_prec"] = precision_score(yte, yte_pred, zero_division=0)
    metrics["test_rec"] = recall_score(yte, yte_pred, zero_division=0)

    metrics["val_auc"] = roc_auc_score(yv, yv_proba)
    metrics["test_auc"] = roc_auc_score(yte, yte_proba)

    return metrics

# Linear model (degree 1) 
linear_model = LogisticRegression(
    max_iter=1*000,
    class_weight="balanced"
)
linear_model.fit(X_train_scaled, y_train)

metrics_linear = evaluate_model(
    linear_model,
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    X_test_scaled, y_test
)
metrics_linear["degree"] = 1

# Polynomial models
degrees = [2, 5, 9]
results = [metrics_linear]
models_info = [("degree=1", linear_model, None)]

for d in degrees:
    print(f"\nTraining degree {d}...")
    poly = PolynomialFeatures(degree=d, include_bias=False)

    Xtr_p = poly.fit_transform(X_train_scaled)
    Xv_p  = poly.transform(X_val_scaled)
    Xte_p = poly.transform(X_test_scaled)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(Xtr_p, y_train)

    m = evaluate_model(model, Xtr_p, y_train, Xv_p, y_val, Xte_p, y_test)
    m["degree"] = d

    results.append(m)
    models_info.append((f"degree={d}", model, poly))

results_df = pd.DataFrame(results)
print("\nAll models:")
print(results_df)

# Pick best model by validation AUC 
best_idx = np.argmax(results_df["val_auc"])
best_label, best_model, best_poly = models_info[best_idx]

print("\nBest model:", best_label)
print(results_df.iloc[best_idx])

# Prepare test set for best model 
if best_poly is None:
    X_test_best = X_test_scaled
else:
    X_test_best = best_poly.transform(X_test_scaled)

y_test_pred = best_model.predict(X_test_best)

# Confusion Matrix 
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred 0", "Pred 1"],
    yticklabels=["True 0", "True 1"]
)
plt.title(f"Confusion Matrix - {best_label}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC curve 
y_test_proba = best_model.predict_proba(X_test_best)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
auc = roc_auc_score(y_test, y_test_proba)

plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC Curve - {best_label}")
plt.legend()
plt.show()
