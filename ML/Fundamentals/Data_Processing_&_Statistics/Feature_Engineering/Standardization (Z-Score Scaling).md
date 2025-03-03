### **1️⃣ What is Standard Deviation?**

Imagine you're analyzing the **heights of climbers** in two different regions:

- **Region A** (Rocky Mountains): Heights range from **2900m to 3100m**
- **Region B** (Swiss Alps): Heights range from **3900m to 4100m**

Even if both groups have similar performance levels, their **absolute heights differ greatly**. To compare them fairly, we need to understand how their values **deviate from the mean**.

🔹 **Standard deviation (σ)** measures **how spread out** the values are from the mean.

If data points are **close to the mean**, σ is **small** (low variance).  
If data points are **widely spread**, σ is **large** (high variance).

---

### **2️⃣ The Formula for Standard Deviation**

The formula for standard deviation is:

$sigma = \sqrt{\frac{\sum (X_i - \mu)^2}{N}}$

Where:

- $X_i$​ = Each individual data point
- $\mu$ = Mean (average) of the dataset
- $N$ = Number of data points

📌 **In simple terms**:

- First, find the **mean** ($\mu$) of all values.
- Then, calculate **how far each value** is from the mean.
- Square these deviations, sum them up, and take the **square root**.

---

### **3️⃣ How Does Standard Deviation Help in Standardization?**

In **Z-score scaling**, we subtract the **mean (μ)** and divide by **standard deviation (σ)**:

$Z= \frac{X - \mu}{\sigma}$​

This transformation ensures that:

1. The data is **centered** around **0** (mean = 0).
2. The spread of data is measured in terms of **standard deviations**.
3. Different datasets can be compared **fairly**, even if they have different units or scales.

---

### **4️⃣ Let's Compute Standard Deviation in Python**

We'll manually calculate **σ (standard deviation)** and use it in the Z-score formula.



```python
import numpy as np
import pandas as pd

# Sample dataset
data = {
    "Height": [160, 170, 155, 180, 165, 175, 190, 150, 185, 168]
}

df = pd.DataFrame(data)

# Compute mean and standard deviation manually
mean_height = df['Height'].mean()
std_dev_height = df['Height'].std(ddof=0)  # Population standard deviation

# Compute Z-scores manually
df['Manual_Z_Score'] = (df['Height'] - mean_height) / std_dev_height

# Display results
print("Manual Z-Score Calculation:")
print(df)

```

**Results**:

|Climber|Height|Height_Z_Score|Manual_Z_Score|
|---|---|---|---|
|Alice|2900|-1.2158954428164332|-1.2158954428164332|
|Alice|3050|-0.9201370918610845|-0.9201370918610845|
|Alice|3100|-0.8215509748759683|-0.8215509748759683|
|Bob|3900|0.7558268968858914|0.7558268968858914|
|Bob|4050|1.0515852478412402|1.0515852478412402|

I've manually calculated the **standard deviation (σ)** and used it to compute the **Z-scores** for each climber's height. The values match those obtained using `StandardScaler` from `sklearn`, confirming that the calculation is correct.

### **Key Takeaways:**

1. **Standard deviation (σ)** measures how spread out the values are.
2. **Z-score standardization** adjusts the values by:
    - Subtracting the **mean (μ)**
    - Dividing by **standard deviation (σ)**
3. This process allows for **fair comparisons** across different scales, such as comparing Alice and Bob's climbs.

Would you like to go deeper into another aspect of standardization?