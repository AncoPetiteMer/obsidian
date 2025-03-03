### **The Tale of the Magic Mirror: Understanding Eigenvalues & Eigenvectors in Machine Learning**

Imagine you're in a **hall of mirrors** at a carnival.

- Some mirrors **stretch you** tall.
- Others **compress** you into a tiny version.
- Some even **tilt you sideways**!

Each mirror represents a **transformation** of your shape—but in all cases, some key features of your **essence remain unchanged**.

In the world of **machine learning**, **Eigenvalues and Eigenvectors** act like those mirrors. They help us **understand transformations** in data and are crucial for methods like **PCA, SVD, and even deep learning**.

---

### **1️⃣ What Are Eigenvalues & Eigenvectors?**

[[Eigenvalues and eigenvectors (linear algebra)]] come from **linear algebra** and are used to **understand how data changes** when transformed.

#### **Eigenvector ($\mathbf{v}$)**

A special **direction** that remains **unchanged** (except for scaling) after a transformation.

#### **Eigenvalue ($\lambda$)**

A **scalar value** that tells how much an eigenvector is **stretched or shrunk** after transformation.

Mathematically:

$A \mathbf{v} = \lambda \mathbf{v}$

Where:

- AAA is a **matrix transformation** (like a dataset in PCA).
- $\mathbf{v}$ is an **eigenvector** (direction that remains unchanged).
- $\lambda$ is an **eigenvalue** (how much that direction is scaled).

📌 **Why is this useful in Machine Learning?**

- Helps **identify key patterns** in data.
- Used in **dimensionality reduction** (PCA).
- Optimizes models by **removing redundant data**.
- Helps in **feature selection** by understanding **variance in data**.

---

### **2️⃣ Eigenvalues & Eigenvectors in PCA**

In **PCA (Principal Component Analysis)**:

- **Eigenvectors** represent the **principal components** (new feature directions).
- **Eigenvalues** tell **how important** each principal component is.

Let’s now apply **Eigenvalues & Eigenvectors in Python** using Alice & Bob’s climbing data. 🚀

```python
import numpy as np
import pandas as pd

# Compute the covariance matrix (measures how features vary together)
cov_matrix = np.cov(climbing_data_scaled.T)  # Transpose to match correct shape

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Create a DataFrame to display the results
eigen_data = pd.DataFrame({
    'Eigenvalue': eigenvalues,
    'Explained Variance (%)': (eigenvalues / np.sum(eigenvalues)) * 100  # Contribution of each component
})

# Display the results
print("Eigenvalues and Explained Variance:")
print(eigen_data)

```



|Eigenvalue|Explained Variance (%)|
|---|---|
|3.576313973883473|99.34205483009644|
|0.02364472982840159|0.6567980507889328|
|4.129628812643625e-05|0.0011471191146232287|

I've computed the **Eigenvalues** and their **Explained Variance** for Alice & Bob’s climbing data.

### **Key Takeaways:**

1. **Eigenvalues indicate the "importance" of each Principal Component (PC)**
    
    - The **first eigenvalue (3.576)** accounts for **99.34% of the total variance**
    - The **second (0.0236) and third (0.000041) eigenvalues** contribute very little
2. **Explained Variance (%) shows how much each PC contributes to data representation**
    
    - The **first PC captures nearly all the information**
    - The remaining PCs contribute **less than 1%**

📌 **What does this mean for Machine Learning?**

- Since **one principal component dominates**, we can **reduce dimensions** while preserving information.
- PCA helps remove **redundant** or **less significant** features, improving **efficiency and accuracy** in ML models.