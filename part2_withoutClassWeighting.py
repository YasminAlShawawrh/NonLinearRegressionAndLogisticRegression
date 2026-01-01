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
)
import matplotlib.pyplot as plt

# Load dataset 
csv_file = "C:/Users/Asus/OneDrive/Documents/ML_Ass2/customer_data.csv"  
TARGET_COLUMN = "ChurnStatus"   

data = pd.read_csv(csv_file)

# handle missing values
# replace missing values in numeric columns with the column mean:
data = data.fillna(data.mean(numeric_only=True))

y_raw = data[TARGET_COLUMN] #Extracts the ChurnStatus column from the DataFrame and stores it as y_raw

if y_raw.dtype == "O":
    y = (y_raw == "Yes").astype(int)
else:
    y = y_raw

# Drop ChurnStatus + ID column from features
X = data.drop(columns=[TARGET_COLUMN, "CustomerID"])

# One-hot encode categorical columns 
X = pd.get_dummies(X, drop_first=True)

print("X shape:", X.shape)
print("Total rows:", len(X))

# Train/val/test split 
total_samples = len(X) #total_samples is how many rows samples we have

# 2500 / 500 / 500
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=2500, random_state=0, stratify=y
) #First split so that train has exactly 2500 samples (train_size=2500) and the rest go into X_temp, y_temp.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp
) #Then split X_temp, y_temp into 50% validation and 50% test.

print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

# Scaling --> Logistic Regression is sensitive to feature scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Helper for metrics 
def evaluate_model_with_proba(model, X_tr, y_tr, X_v, y_v, X_te, y_te):
    metrics = {}
    # 0/1 predictions
    y_tr_pred = model.predict(X_tr)
    y_v_pred = model.predict(X_v)
    y_te_pred = model.predict(X_te)

    # probabilities for ROC / AUC
    y_v_proba = model.predict_proba(X_v)[:, 1]
    y_te_proba = model.predict_proba(X_te)[:, 1]

    metrics["train_acc"] = accuracy_score(y_tr, y_tr_pred)
    metrics["train_prec"] = precision_score(y_tr, y_tr_pred, zero_division=0)
    metrics["train_rec"] = recall_score(y_tr, y_tr_pred, zero_division=0)

    metrics["val_acc"] = accuracy_score(y_v, y_v_pred)
    metrics["val_prec"] = precision_score(y_v, y_v_pred, zero_division=0)
    metrics["val_rec"] = recall_score(y_v, y_v_pred, zero_division=0)

    metrics["test_acc"] = accuracy_score(y_te, y_te_pred)
    metrics["test_prec"] = precision_score(y_te, y_te_pred, zero_division=0)
    metrics["test_rec"] = recall_score(y_te, y_te_pred, zero_division=0)

    metrics["val_auc"] = roc_auc_score(y_v, y_v_proba)
    metrics["test_auc"] = roc_auc_score(y_te, y_te_proba)

    return metrics

# Linear logistic regression (degree 1)
linear_model = LogisticRegression(max_iter=1000) #Creates a logistic regression model named linear_model.
linear_model.fit(X_train_scaled, y_train)

metrics_linear = evaluate_model_with_proba( #Calls the helper function to compute all metrics for this model.
    linear_model,
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    X_test_scaled, y_test
)
metrics_linear["degree"] = 1

print("\nLinear model metrics:")
print(metrics_linear)

# Polynomial degrees 2, 5, 9 
degrees = [2, 5, 9] #list of polynomial degrees
results = [metrics_linear]
models_info = [("degree=1", linear_model, None)]

for d in degrees:
    print(f"\nTraining polynomial degree {d} model...")
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled) #Transforms X_train_scaled into polynomial features
    X_val_poly = poly.transform(X_val_scaled) #Transforms X_val_scaled into polynomial features
    X_test_poly = poly.transform(X_test_scaled) #Transforms X_test_scaled into polynomial features

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_poly, y_train)

    metrics_d = evaluate_model_with_proba(
        model,
        X_train_poly, y_train,
        X_val_poly, y_val,
        X_test_poly, y_test
    )
    metrics_d["degree"] = d
    results.append(metrics_d)
    models_info.append((f"degree={d}", model, poly))

results_df = pd.DataFrame(results)
print("\nAll models:")
print(results_df)

# Pick best model by validation AUC
best_idx = np.argmax(results_df["val_auc"].values)
best_label, best_model, best_poly = models_info[best_idx]
print(f"\nBest model: {best_label}")
print(results_df.iloc[best_idx])

# ROC curve on test set 
if best_poly is None:
    X_test_best = X_test_scaled
else:
    X_test_best = best_poly.transform(X_test_scaled)

y_test_proba = best_model.predict_proba(X_test_best)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
auc = roc_auc_score(y_test, y_test_proba)

plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({best_label})")
plt.legend()
plt.show()
