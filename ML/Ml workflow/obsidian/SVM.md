### **Storytelling Approach to Understanding SVM (Support Vector Machine)**

#### **The Tale of the Fruit Vendor 🍎🍊**

Once upon a time, in a small town, there was a fruit vendor named Sam. Sam sold two types of fruits: apples 🍎 and oranges 🍊. However, a new helper, Jake, joined his shop, and he couldn't tell the difference between apples and oranges.

To help Jake, Sam decided to **classify the fruits** based on two features:

1. **Weight** (grams)
2. **Size/Diameter** (cm)

Sam plotted the fruits on a graph with:

- **Apples** represented as red dots (🔴).
- **Oranges** represented as blue dots (🔵).

Jake noticed that apples and oranges seemed to form two separate groups, but he needed a rule to separate them. Sam came up with an idea:  
_"Let's draw a line that best separates the apples from the oranges."_

But here was the problem—there were multiple ways to draw a line. Some lines would misclassify a few apples as oranges or vice versa. Sam wanted the **best** possible line.

### **Introducing SVM (Support Vector Machine)**

SVM helped Sam by finding the **optimal decision boundary** (the best line). This boundary:

1. **Maximizes the margin** (the distance between the line and the nearest points of each fruit type).
2. **Uses support vectors** (fruits closest to the boundary that help define the margin).

This way, even if a new fruit arrives, the boundary helps classify whether it is an apple or an orange based on its weight and size.

---

### **Python Example of SVM (Classifying Fruits)**

Now, let's see how we can classify apples and oranges using SVM in Python.

#### **Step 1: Import Libraries & Generate Data**

We'll use `sklearn` to build an SVM model.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (Weight in grams, Size in cm)
X = np.array([[150, 7], [160, 7.5], [170, 8], [140, 6.5], [130, 6],  # Apples
              [200, 9], [210, 10], [220, 10.5], [190, 9], [180, 8.5]]) # Oranges

# Labels: 0 - Apple, 1 - Orange
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear')  # Using linear kernel
svm_model.fit(X_train, y_train)

# Predict on test set
y_pred = svm_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Plot Decision Boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel("Weight (g)")
    plt.ylabel("Size (cm)")
    plt.title("SVM Decision Boundary")
    plt.show()

plot_decision_boundary(svm_model, X, y)

```

---

### **Understanding the Output**

1. **Model Accuracy**: We check how well our model classifies fruits.
2. **Decision Boundary Plot**:
    - The SVM model draws a **straight line** (since we used a `linear` kernel).
    - The fruits are separated by this line, ensuring correct classification.
    - The **support vectors** (closest points to the boundary) define the margin.

---

### **Conclusion**

With SVM, Sam successfully trained Jake to classify apples and oranges based on weight and size. Now, even if a new fruit arrives, the model can confidently say if it's an apple or an orange. 🎉

This is how **Support Vector Machine (SVM)** helps in classification problems in machine learning! 🚀