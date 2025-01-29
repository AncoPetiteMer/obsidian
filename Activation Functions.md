### **What are Activation Functions?**

Imagine you’re a teacher trying to grade a class assignment. You collect all the raw scores from your students, but the scores are just numbers—some are too high, some are too low, and they don’t tell you how well a student performed overall. To make sense of it, you transform the raw scores into meaningful grades, like A, B, C, and so on, or into percentages out of 100. This transformation helps you interpret the scores more effectively.

In machine learning, **activation functions** play a similar role. They take the raw outputs (weighted sums) from neurons in a neural network and transform them into meaningful values that the network can use to learn complex patterns. Without activation functions, a neural network would just be a stack of linear equations, which limits its ability to model non-linear relationships (like predicting fraud or recognizing images).

---

### **The Story: Fraud Detection with Neural Networks**

Imagine you’re building a neural network to detect fraudulent credit card transactions. The raw inputs (like `Amount`, `Time`, and `Location`) are processed through the network’s layers, where each neuron computes a weighted sum of its inputs. But the outputs of these neurons are just raw numbers—they could be positive, negative, or very large, and they’re not easy to interpret.

Here’s where activation functions come in. By applying an activation function to each neuron’s output:

1. You can introduce **non-linearity**, helping the network learn complex patterns in the data (like identifying clusters of fraud).
2. You can squash the outputs into a specific range, making them easier to work with.

For example:

- A **ReLU activation function** turns negative outputs into 0 and keeps positive ones unchanged. This makes the network more efficient.
- A **sigmoid activation function** squashes outputs into a range between 0 and 1, which is useful for predicting probabilities (e.g., the probability of fraud).

---

### **Step 1: Why Do We Need Activation Functions?**

1. **Introducing Non-Linearity**:
    
    - Real-world problems (like fraud detection) often involve non-linear relationships.
    - Without activation functions, the network is just a linear transformation, no matter how many layers it has.
2. **Squashing Outputs**:
    
    - Activation functions can squash large outputs into manageable ranges (e.g., between 0 and 1 or -1 and 1).
    - This makes the network’s outputs easier to interpret and prevents exploding values.
3. **Learning Complex Patterns**:
    
    - Activation functions enable the network to learn more complex, non-linear patterns in the data.

---

### **Step 2: Common Activation Functions**

Here are some of the most commonly used activation functions:

#### **1. Sigmoid**

- Squashes the output into a range between 0 and 1.
- Often used in the output layer for binary classification (e.g., fraud vs. non-fraud).
- Formula: 
$σ(x)=11+e−x\sigma(x) = \frac{1}{1 + e^{-x}}σ(x)=1+e−x1​$
- **Example**: If the output is 0.8, it could represent an 80% chance of fraud.

---

#### **2. ReLU (Rectified Linear Unit)**

- Sets negative values to 0 and leaves positive values unchanged.
- Commonly used in hidden layers of neural networks.
- Formula:
-$f(x)=max⁡(0,x)f(x) = \max(0, x)f(x)=max(0,x)$
- **Example**: If the neuron output is -5, ReLU converts it to 0. If it’s 10, it remains 10.

---

#### **3. Tanh (Hyperbolic Tangent)**

- Squashes the output into a range between -1 and 1.
- Useful when you want outputs with both positive and negative ranges.
- Formula:
- $tanh⁡(x)=ex−e−xex+e−x\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}tanh(x)=ex+e−xex−e−x​$
- **Example**: An output of 0.5 indicates moderate activation, while -0.5 indicates moderate suppression.

---

#### **4. Softmax**

- Converts raw outputs into probabilities that sum to 1.
- Used in the output layer for multi-class classification.
- Formula: $softmax(xi)=exi∑jexj\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}softmax(xi​)=∑j​exj​exi​​$
- **Example**: If the network predicts `[2.0, 1.0, 0.1]`, softmax transforms it into `[0.7, 0.2, 0.1]`, which can be interpreted as probabilities.

---

### **Step 3: Python Examples**

#### **1. Visualizing Activation Functions**

Let’s plot some common activation functions to understand how they work.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Input range
x = np.linspace(-10, 10, 100)

# Plot activation functions
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid(x), label="Sigmoid", color="blue")
plt.plot(x, relu(x), label="ReLU", color="green")
plt.plot(x, tanh(x), label="Tanh", color="red")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.show()

```

---

#### **2. Using Activation Functions in PyTorch**

Let’s build a simple neural network and apply activation functions to its layers.

```python
import torch
import torch.nn as nn

# Define a neural network class
class FraudDetectionNN(nn.Module):
    def __init__(self):
        super(FraudDetectionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 16),  # Input layer (3 features) to hidden layer (16 neurons)
            nn.ReLU(),         # Apply ReLU activation
            nn.Linear(16, 1),  # Hidden layer to output layer (1 neuron)
            nn.Sigmoid()       # Apply Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.network(x)

# Initialize the model
model = FraudDetectionNN()

# Example input: [Amount, Time, Location]
X = torch.tensor([[5000.0, 3600.0, 1.0], [100.0, 7200.0, 0.0]], dtype=torch.float32)

# Forward pass
output = model(X)

# Display the output
print("Model Output (Fraud Probability):")
print(output)

```


**Explanation**:

1. The **ReLU activation** in the hidden layer helps the network learn non-linear patterns.
2. The **Sigmoid activation** in the output layer converts the raw score into a probability (e.g., 0.8 = 80% chance of fraud).

---

### **Step 4: Why Choose Different Activation Functions?**

1. **Hidden Layers**:
    
    - Use **ReLU** for efficiency and to avoid the vanishing gradient problem.
    - Use **Tanh** when you want outputs in a range that includes negatives (e.g., [-1, 1]).
2. **Output Layers**:
    
    - Use **Sigmoid** for binary classification (e.g., fraud vs. non-fraud).
    - Use **Softmax** for multi-class classification.
3. **Specific Use Cases**:
    
    - Use **Leaky ReLU** (a variation of ReLU) if you want to avoid "dead neurons" (neurons that output 0 for all inputs).

---

### **Step 5: When to Use Activation Functions**

1. **Binary Classification**:
    
    - Use **Sigmoid** in the output layer to get probabilities.
2. **Multi-Class Classification**:
    
    - Use **Softmax** in the output layer to assign probabilities to each class.
3. **Hidden Layers**:
    
    - Use **ReLU** for faster convergence and better performance.

---

### **Why Are Activation Functions Important?**

1. **Introduce Non-Linearity**:
    
    - Without activation functions, neural networks can only learn linear relationships.
2. **Control Outputs**:
    
    - Activation functions squash outputs into a manageable range, preventing issues like exploding gradients.
3. **Enable Learning**:
    
    - By transforming raw outputs, activation functions make it possible for the network to learn complex patterns in the data.

---

### **Key Takeaway**

**Activation Functions** are like the translators in a neural network, transforming raw outputs into meaningful signals that help the network learn complex, non-linear patterns. Whether it’s ReLU for hidden layers or Sigmoid for binary outputs, activation functions are essential for making neural networks powerful and versatile. ⚡✨