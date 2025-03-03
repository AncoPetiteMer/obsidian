Bootstrap sampling is a powerful statistical technique used in machine learning and statistics. In this tutorial, we'll dive deep into what bootstrap sampling is, how it works, and its applications. We'll also implement it step-by-step in Python to understand it fully.

---

## **1. What is Bootstrap Sampling?**

**Bootstrap sampling** is a technique where we randomly draw samples **with replacement** from a dataset to create multiple subsets. It is commonly used to:

- **Estimate statistics** (e.g., mean, variance, confidence intervals) when the dataset is small.
- **Create diverse datasets** for ensemble learning methods like **[[Random Forest]]**([[Titanic Survival Prediction Story 🚢🌲]])

---

### **1.1 How Does Bootstrap Sampling Work?**

1. From a dataset of size N, we randomly sample N data points **with replacement**.
2. This means that:
    - Some data points will appear multiple times in the sample.
    - Some data points may not appear at all.
3. The process is repeated to create multiple bootstrap samples.

---

### **2. Why Use Bootstrap Sampling?**

Bootstrap sampling has several benefits:

1. **Diversity**: Creates diverse subsets of the data for training models.
2. **Stability**: Reduces overfitting by aggregating predictions from different subsets.
3. **Out-of-Bag Evaluation**: Data points not selected in a sample can be used for validation.

---

### **3. Example Scenario**

Imagine we have the following dataset of students' scores:

`Original Dataset: [85, 90, 78, 92, 88]`

#### **Step 1**: Bootstrap Sample 1

Randomly sample 5 data points (with replacement):


`Sample 1: [90, 85, 78, 92, 85]`

#### **Step 2**: Bootstrap Sample 2

Another random sample:

`Sample 2: [85, 88, 88, 92, 90]`

Notice:

- **With Replacement**: Some values appear multiple times.
- **Diversity**: Each sample is slightly different.

---

## **4. Python Implementation**

---

### **4.1 Manual Bootstrap Sampling**

Let’s implement bootstrap sampling manually to understand the process.

```python
import numpy as np

# Original dataset: Student scores
data = np.array([85, 90, 78, 92, 88])

# Number of bootstrap samples to generate
num_samples = 3

# Generate bootstrap samples
for i in range(num_samples):
    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    print(f"Bootstrap Sample {i+1}: {bootstrap_sample}")

```

#### **Output**:

```python
Bootstrap Sample 1: [88 92 85 92 85]
Bootstrap Sample 2: [90 88 78 85 85]
Bootstrap Sample 3: [90 88 88 92 90]

```
---

### **4.2 Bootstrap Sampling with Pandas**

We can use **Pandas** to work with tabular datasets and create bootstrap samples.

```python
import pandas as pd

# Create a small dataset
data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Score': [85, 90, 78, 92, 88]
})

# Generate a bootstrap sample
bootstrap_sample = data.sample(n=len(data), replace=True, random_state=42)
print("Original Dataset:")
print(data)
print("\nBootstrap Sample:")
print(bootstrap_sample)

```

#### **Output**:

```python
Original Dataset:
      Name  Score
0    Alice     85
1      Bob     90
2  Charlie     78
3    David     92
4      Eva     88

Bootstrap Sample:
      Name  Score
1      Bob     90
4      Eva     88
2  Charlie     78
0    Alice     85
4      Eva     88

```

---

### **4.3 Multiple Bootstrap Samples**

Let’s create **multiple bootstrap samples** programmatically.

```python
# Number of bootstrap samples to generate
num_samples = 3

# Generate multiple bootstrap samples
for i in range(num_samples):
    bootstrap_sample = data.sample(n=len(data), replace=True, random_state=i)
    print(f"\nBootstrap Sample {i+1}:")
    print(bootstrap_sample)

```

---

### **4.4 Out-of-Bag (OOB) Samples**

In each bootstrap sample, some data points are **left out**. These are called **out-of-bag (OOB)** samples and can be used for validation.

```python
# Create a bootstrap sample
bootstrap_sample = data.sample(n=len(data), replace=True, random_state=42)

# Identify OOB samples
oob_samples = data.loc[~data.index.isin(bootstrap_sample.index)]

print("Bootstrap Sample:")
print(bootstrap_sample)

print("\nOut-of-Bag (OOB) Samples:")
print(oob_samples)

```

---

## **5. Real-World Use Case: Random Forest**

In Random Forest, bootstrap sampling is used to:

1. Train each decision tree on a different bootstrap sample.
2. Use OOB samples to evaluate the model's performance.

Let’s build a Random Forest using bootstrap sampling on a small dataset.

---

### **5.1 Random Forest with Bootstrap**

We’ll use **Scikit-Learn’s RandomForestClassifier** to demonstrate how bootstrap sampling works in Random Forest.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest with bootstrap sampling
rf_model = RandomForestClassifier(n_estimators=100, bootstrap=True, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

```

---

### **5.2 Feature Importance in Random Forest**

Random Forest uses bootstrap sampling to train trees on different subsets of the data. Let’s inspect the **feature importance**.

```python
import matplotlib.pyplot as plt

# Get feature importance
feature_importances = rf_model.feature_importances_

# Plot feature importance
plt.bar(iris.feature_names, feature_importances, color='teal')
plt.title("Feature Importance in Random Forest")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()

```

---

## **6. Key Takeaways**

1. **Bootstrap Sampling**:
    - Draws random samples **with replacement**.
    - Ensures each model sees a different subset of the data.
2. **Out-of-Bag (OOB) Samples**:
    - Data points not selected in a bootstrap sample can be used for validation.
3. **Applications**:
    - **Random Forest**: Combines bootstrap sampling and aggregation for powerful predictions.
    - **Statistical Estimation**: Estimate confidence intervals and variances.


