

### **Mathematical Foundations of Random Forests**

#### **1. Gini Impurity: Measuring Node Purity**

The Gini Impurity quantifies the probability of incorrectly classifying a randomly chosen element in a node.

where:

- is the proportion of instances of class in the node.
    
- is the number of classes.
    

Example Calculation: If a node has 60% class A and 40% class B:

A **Gini Impurity of 0** means perfect classification.

---

#### **2. Entropy: Measuring Information Gain**

Entropy quantifies uncertainty in the dataset.

For a node with two classes, A (60%) and B (40%):

Entropy is used to compute **Information Gain**:

A higher Information Gain means a better split.

---

#### **3. Bootstrapping: Sampling with Replacement**

Each tree in a Random Forest is trained on a **bootstrap sample** (random sampling with replacement).

If a dataset has observations, each tree is trained on approximately unique samples (since ).

The remaining **out-of-bag (OOB) samples** are used for validation.

---

#### **4. Feature Selection in Random Forest**

At each split, the algorithm randomly selects features from the total features, where:

This ensures **decorrelation** between trees, reducing overfitting.

---

#### **5. Aggregation of Predictions**

Each tree predicts independently, and results are aggregated:

- **Classification:** Majority voting:
    
- **Regression:** Averaging:
    

where is the prediction of tree and is the number of trees.

---

#### **6. Feature Importance Calculation**

The importance of a feature is calculated as the **total Gini decrease** across all trees:

A higher score means the feature is more critical for predictions.

---

These mathematical principles ensure that Random Forests are **robust, accurate, and interpretable** for classification and regression tasks.