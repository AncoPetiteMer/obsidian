### **What is Hyperparameter Tuning?**

Imagine you’re baking a cake, but you’re not following a strict recipe. Instead, you’re experimenting with how much flour, sugar, and butter to use. You might ask questions like:

- Should I add more sugar for sweetness?
- What’s the right oven temperature to bake it perfectly?
- Should I use a larger pan for a fluffier cake?

Each of these decisions affects the outcome of the cake. In the same way, **hyperparameter tuning** in machine learning is about experimenting with and adjusting the "ingredients" of your model to get the best possible results.

---

### **The Story: Predicting House Prices**

You’ve built a machine learning model to predict house prices, but you notice that it’s not performing very well. The problem isn’t with the data itself or the model architecture—it’s with the **hyperparameters** (adjustable settings) of your model. These hyperparameters determine how the model learns and processes the data.

For example:

- **Learning Rate**: How big are the model’s steps when it adjusts itself during training?
- **Number of Neurons**: How many "brain cells" should the model have in its hidden layers?
- **[[Batch Size]]**: Should the model process the data in small groups (mini-batches) or all at once?

The challenge is figuring out which combination of these settings will make your model perform the best.

---

### **Step 1: What are Hyperparameters?**

Hyperparameters are settings you define **before training** a model. Unlike weights and biases (which are learned during training), hyperparameters are **external to the model** and control how it learns.

Examples of hyperparameters include:

1. **[[Learning Rate]]**: How quickly the model adjusts its weights to minimize loss.
2. **[[Batch Size]]**: The number of training samples the model processes at a time.
3. **Number of Layers and Neurons**: Controls the model’s architecture.
4. **Dropout Rate**: Prevents overfitting by randomly disabling neurons during training.
5. **Number of Epochs**: How many full passes through the dataset the model makes during training.

---

### **Step 2: Why is Hyperparameter Tuning Important?**

Imagine you’re driving a car but don’t know the speed limit. If you go too slow (small learning rate), you’ll never reach your destination. If you go too fast (large learning rate), you risk crashing. Similarly, choosing the wrong hyperparameters can:

- Cause the model to train too slowly or overshoot the optimal solution.
- Lead to **overfitting** (model memorizes the training data) or **underfitting** (model fails to learn the patterns) [[Overfitting and Underfitting]].

Hyperparameter tuning is the process of finding the **sweet spot** for these settings to maximize the model’s performance.

---

### **Step 3: Methods for Hyperparameter Tuning**

#### **1. Grid Search**

- Exhaustively searches through a predefined set of hyperparameter values.
- Example: Try all combinations of learning rates and batch sizes.

#### **2. Random Search**

- Randomly samples hyperparameter values within a specified range.
- Faster than Grid Search and works well for large hyperparameter spaces.

#### **3. Bayesian Optimization (Optuna)**

- A smarter approach that learns from past results to suggest the next best hyperparameter values.
- Faster and more efficient than Grid or Random Search.

---

### **Step 4: Python Code for Hyperparameter Tuning**

Let’s tune the hyperparameters of a neural network to improve its performance.

#### **Step 4.1: Setup the Dataset**

We’ll use a simple dataset to predict house prices based on size.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Sample dataset: House sizes (input) and prices (target)
X = torch.tensor([[1.5], [2.0], [2.5], [3.0]], dtype=torch.float32)  # House sizes (1000 sqft)
y = torch.tensor([[300], [400], [500], [600]], dtype=torch.float32)  # Prices (1000s of dollars)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

```
#### **Step 4.2: Define the Model**

```python
import torch.nn as nn

class HousePriceModel(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(HousePriceModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),  # Input layer to hidden layer
            nn.ReLU(),                  # Activation function
            nn.Dropout(dropout_rate),   # Dropout layer for regularization
            nn.Linear(hidden_size, 1)   # Hidden layer to output layer
        )

    def forward(self, x):
        return self.network(x)

```

---

#### **Step 4.3: Hyperparameter Tuning with Grid Search**

```python
import torch.nn as nn
import torch.optim as optim

# Define possible hyperparameter values
hidden_sizes = [8, 16, 32]
learning_rates = [0.01, 0.001]
dropout_rates = [0.1, 0.2]
batch_sizes = [1, 2]
num_epochs = 50

# Track the best configuration
best_loss = float('inf')
best_config = None

# Grid search
for hidden_size in hidden_sizes:
    for learning_rate in learning_rates:
        for dropout_rate in dropout_rates:
            for batch_size in batch_sizes:
                # Define the model
                model = HousePriceModel(hidden_size, dropout_rate)

                # Define the loss function and optimizer
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Train the model
                for epoch in range(num_epochs):
                    for i in range(0, len(X_train), batch_size):
                        batch_X = X_train[i:i+batch_size]
                        batch_y = y_train[i:i+batch_size]

                        # Forward pass
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Evaluate on the test set
                with torch.no_grad():
                    test_outputs = model(X_test)
                    test_loss = criterion(test_outputs, y_test)

                # Update the best configuration
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_config = {
                        "hidden_size": hidden_size,
                        "learning_rate": learning_rate,
                        "dropout_rate": dropout_rate,
                        "batch_size": batch_size
                    }

print("Best Configuration:", best_config)
print("Best Loss:", best_loss.item())

```

---

#### **Step 4.4: Hyperparameter Tuning with Optuna (Bayesian Optimization)**

Instead of manually trying combinations, let’s use **Optuna** for smarter tuning.

```python
import optuna
import torch.nn as nn
import torch.optim as optim

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 8, 32)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_int('batch_size', 1, 4)

    # Define the model
    model = HousePriceModel(hidden_size, dropout_rate)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on the test set
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)

    return test_loss.item()

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Best hyperparameters
print("Best Hyperparameters:", study.best_params)

```



---

### **Why is Hyperparameter Tuning Important?**

Without tuning:

1. Your model might underperform, no matter how good your data is.
2. You might waste time with trial-and-error experiments.
3. You risk overfitting or underfitting.

Hyperparameter tuning ensures your model is trained efficiently and performs its best.

---

### **Key Takeaway**

**Hyperparameter Tuning** is like perfecting a recipe—you adjust the ingredients (hyperparameters) to get the best results. Whether you use Grid Search, Random Search, or Bayesian Optimization, tuning is essential for creating a high-performing machine learning model. 🍰✨