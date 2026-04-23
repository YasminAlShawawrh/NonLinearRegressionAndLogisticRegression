# Non-Linear Regression & Logistic Regression

A machine learning assignment implementing non-linear regression techniques and binary classification using logistic regression. The project is split into three parts covering ridge regression, RBF basis functions, and customer churn prediction with polynomial decision boundaries.

---

## Table of contents

- [Project overview](#project-overview)
- [Part 1 — Non-linear regression](#part-1--non-linear-regression)
- [Part 2 — Logistic regression (basic)](#part-2--logistic-regression-basic)
- [Part 3 — Logistic regression (with confusion matrix)](#part-3--logistic-regression-with-confusion-matrix)
- [Dataset](#dataset)
- [Technologies used](#technologies-used)
- [How to run](#how-to-run)

---

## Project overview

This project explores how different regression and classification techniques handle non-linearity and generalization. It compares models of varying complexity and evaluates them using standard ML metrics including accuracy, precision, recall, AUC, ROC curves, and confusion matrices.

---

## Part 1 — Non-linear regression

Fits a noisy sine function `y = sin(5πx) + noise` using two approaches.

### A — Ridge regression with polynomial features (degree 9)

Trains ridge regression models across 5 different regularization strengths to observe the bias-variance tradeoff:

| λ value | Effect |
|---|---|
| 0 | No regularization — high variance (overfitting) |
| 0.0000001 | Very slight regularization |
| 0.0001 | Mild smoothing |
| 0.5 | Strong regularization |
| 5 | Heavy regularization — high bias (underfitting) |

Plots all fits alongside the true function to visualize how λ controls model complexity.

### B — RBF (Radial Basis Function) regression

Fits the same dataset using Gaussian basis functions with increasing numbers of RBF centers:

| M (# of RBFs) | Behavior |
|---|---|
| 1 | Too simple — underfits |
| 5 | Moderate fit |
| 10 | Good approximation |
| 50 | Very flexible — risk of overfitting |

Each RBF is a Gaussian bump centered at equally spaced points, with width `σ = 1/M`. Linear regression is applied on top of the RBF feature matrix.

---

## Part 2 — Logistic regression (basic)

Binary classification on the customer churn dataset using logistic regression with linear and polynomial features.

**Pipeline:**
- Loads `customer_data.csv` and imputes missing values with column means
- Encodes the `ChurnStatus` target as 0/1
- One-hot encodes categorical features
- Splits data into train / validation / test (2500 / 500 / 500)
- Applies `StandardScaler` before fitting
- Trains logistic regression at degrees 1, 2, 5, and 9
- Selects the best model by **validation AUC**
- Plots the ROC curve for the best model on the test set

**Metrics tracked per model:** accuracy, precision, recall, AUC (validation + test)

---

## Part 3 — Logistic regression (with confusion matrix)

Same pipeline as Part 2 with one key addition — uses `class_weight="balanced"` to handle class imbalance in the churn dataset, and adds a **confusion matrix heatmap** for the best model.

**Additional output:**
- Confusion matrix plotted with `seaborn` heatmap
- ROC curve with AUC score on test set

---

## Dataset

- **File:** `customer_data.csv`
- **Target:** `ChurnStatus` (binary — churn / no churn)
- **ID column:** `CustomerID` (dropped before training)
- **Split:** 2500 train · 500 validation · 500 test

---

## Technologies used

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations, data generation |
| `pandas` | Data loading and preprocessing |
| `scikit-learn` | Models, scaling, metrics, polynomial features |
| `matplotlib` | All plots and visualizations |
| `seaborn` | Confusion matrix heatmap (Part 3) |

---

## How to run

1. Clone the repository:
```bash
git clone https://github.com/YasminAlShawawrh/NonLinearRegressionAndLogisticRegression.git
cd NonLinearRegressionAndLogisticRegression
```

2. Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

3. Update the dataset path in Parts 2 and 3:
```python
csv_file = "path/to/your/customer_data.csv"
```

4. Run each part individually:
```bash
python part1.py   # Ridge & RBF regression
python part2.py   # Basic logistic regression
python part3.py   # Logistic regression + confusion matrix
```
