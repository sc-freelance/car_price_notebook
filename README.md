# car_price_notebook

# Vehicle Regression Engine — From Scratch & Scikit-Learn

This project provides an end-to-end mathematical exploration of **Linear**, **Polynomial**, and **Logistic Regression** architectures applied directly to vehicle asset data (`CarPrice_Assignment.csv`). 

The core mission of this repository is to move past treating machine learning frameworks as "black boxes" by dissecting how feature variables interact, mappings compute over specialized non-linear intervals, and errors minimize at an algorithmic level.

---

## 📈 Implemented Methodologies & Mechanics

### 1. Simple Linear Regression
* **Objective:** Map structural pricing trends on a continuous linear spectrum against engine performance traits (`price` vs `horsepower`).
* **Mechanics:** Partitioned data dynamically using an `80/20` evaluation split, optimizing model tracking coefficients over continuous target labels.

### 2. High-Degree Polynomial Regression
* **Objective:** Capture complex, non-linear market pricing fluctuations where a straight-line fit fails to represent data tendencies accurately.
* **Mechanics:** Extrapolated performance inputs into multi-dimensional feature space using an escalated 5th-degree polynomial function mapping ($y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_5 x^5$). Employs custom space sorting via NumPy (`linspace`) to draw smooth, high-fidelity trend lines over noisy scattered inputs.

### 3. Logistic Regression from Scratch (Theoretical Implementation)
* **Objective:** Understand probability and classification bounds without external packages.
* **Core Concept:** Combines input features into a linear equation, wraps them inside a continuous Sigmoid activation function ($S(z) = \frac{1}{1 + e^{-z}}$) to squeeze evaluations strictly between $0$ and $1$, and trains iteratively via Gradient Descent optimization loops.

---

## 📊 Visualizations

The project generates clean diagnostic visual plots via Matplotlib to contrast true values with our predicted lines:

* **Data Profiling:** Initial scatter tracking plots checking structural correlations and filtering null data points.
* **Linear vs Curve Fit:** Comparative charts revealing the structural differences between standard linear regression lines and advanced high-degree polynomial curves.
