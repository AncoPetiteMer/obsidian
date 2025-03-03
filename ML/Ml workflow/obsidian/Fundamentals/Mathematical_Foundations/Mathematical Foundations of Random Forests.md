### **Mathematical Foundations of Random Forests**

#### **1. [[Gini Impurity]]: Measuring Node Purity**

The Gini Impurity quantifies the probability of incorrectly classifying a randomly chosen element in a node.

$Gini = 1 - \sum_{i=1}^{k} p_i^2$

where:

- pip_i is the proportion of instances of class ii in the node.
- kk is the number of classes.

Example Calculation: If a node has 60% class A and 40% class B:

$Gini = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48$

A **Gini Impurity of 0** means perfect classification.

---

#### **2. Entropy: Measuring Information Gain**

Entropy quantifies uncertainty in the dataset.

$Entropy = - \sum_{i=1}^{k} p_i \log_2 p_i$

For a node with two classes, A (60%) and B (40%):

$Entropy = - (0.6 \log_2(0.6) + 0.4 \log_2(0.4)) \approx 0.971$

Entropy is used to compute **Information Gain**:

$IG = Entropy_{parent} - \sum_{children} \frac{n_{child}}{n_{parent}} \times Entropy_{child}$

A higher Information Gain means a better split.

---

#### **3. Bootstrapping: Sampling with Replacement**

Each tree in a Random Forest is trained on a **bootstrap sample** (random sampling with replacement).

If a dataset has nn observations, each tree is trained on approximately 0.632n0.632n unique samples (since 1−e−1≈0.6321 - e^{-1} \approx 0.632).

The remaining **out-of-bag (OOB) samples** are used for validation.

---

#### **4. Feature Selection in Random Forest**

At each split, the algorithm randomly selects mm features from the total MM features, where:

$m = \sqrt{M} \text{ (for classification)}$
$m = \frac{M}{3} \text{ (for regression)}$

This ensures **decorrelation** between trees, reducing overfitting.

---

#### **5. Aggregation of Predictions**

Each tree predicts independently, and results are aggregated:

- **Classification:** Majority voting:
    
    $\hat{y} = \operatorname{mode}(y_1, y_2, ..., y_T)$
- **Regression:** Averaging:
    
    $\hat{y} = \frac{1}{T} \sum_{t=1}^{T} y_t$

where yty_t is the prediction of tree tt and TT is the number of trees.

---

#### **6. Feature Importance Calculation**

The importance of a feature ff is calculated as the **total Gini decrease** across all trees:

$Importance(f) = \sum_{splits} \left( Gini_{parent} - Weighted \ Gini_{children} \right)$

A higher score means the feature is more critical for predictions.

---

These mathematical principles ensure that Random Forests are **robust, accurate, and interpretable** for classification and regression tasks.