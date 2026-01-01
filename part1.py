import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression



# Generate dataset S
np.random.seed(100) # makes the random numbers repeatable

n = 25 # number of data points
x = np.random.uniform(0, 1, n) # generates 25 random x-values between 0 and 1
noise = np.random.uniform(-0.3, 0.3, n) # generate random noise for each point, between –0.3 and +0.3
y = np.sin(5 * np.pi * x) + noise # generates the y values using the formula= sin(5πx) + noise

xs = np.linspace(0, 1, 400) # makes smooth curves
ys_true = np.sin(5 * np.pi * xs) # computes the true function without noise 

# part A #
# Polynomial features (degree = 9)
degree = 9 # sets the polynomial degree = 9
poly = PolynomialFeatures(degree) # creates a PolynomialFeatures (This can make curves, waves, and smooth shapes, not just straight lines)object
X = poly.fit_transform(x.reshape(-1, 1)) # Converts x-values into polynomial form:[1, x, x², x³, …, x¹⁰]
Xs = poly.transform(xs.reshape(-1, 1))

# Ridge regression for different λ
lambdas = [0, 0.0000001, 0.0001, 0.5, 5] #List of λ (regularization strengths) to compare
plt.figure(figsize=(2, 7))
plt.scatter(x, y, color='black', label="Data Points") #Plots the original noisy data points

for lam in lambdas: #Loop over each λ value
    model = Ridge(alpha=lam) #Creates a Ridge model with regularization strength λ
    model.fit(X, y) #Trains the model using the polynomial features
    y_pred = model.predict(Xs) #Predicts values on the smooth xs grid
    plt.plot(xs, y_pred, label=f"lambda = {lam}")

plt.plot(xs, ys_true, 'k--', linewidth=2, label="True function")
plt.title("Ridge Regression with Different λ Values")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# part B #
# RBF function
# Computes a Gaussian "bump" centered at c with width sigma
# x: input points c: center of the RBF sigma: controls the width (spread) of the RBF
def rbf(x, c, sigma):
    return np.exp(-(x - c)**2 / (2 * sigma**2))

rbf_counts = [1, 5, 10, 50] # Different numbers of RBFs

plt.figure(figsize=(12, 10))

for i, M in enumerate(rbf_counts, 1):
    centers = np.linspace(0, 1, M)
    # Choose RBF width (sigma): smaller when we have more RBFs
    sigma = 1 / (M)  # reasonable width

    # Construct RBF feature matrix
    Phi = np.vstack([rbf(x, c, sigma) for c in centers]).T
    Phi_s = np.vstack([rbf(xs, c, sigma) for c in centers]).T

    # Linear regression
    model = LinearRegression()
    model.fit(Phi, y)
    y_pred = model.predict(Phi_s)

    plt.subplot(2, 2, i)
    plt.scatter(x, y, color='black') # noisy data points
    plt.plot(xs, ys_true, 'k--', label="True Function") # true sin curve
    plt.plot(xs, y_pred, label=f"{M} RBFs") # model prediction
    plt.title(f"RBF Regression with {M} Basis Functions")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
