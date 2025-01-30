### **Naïve Bayes – The Probability-Based Classifier**

> _Imagine you are a detective solving a mystery. You have different clues about the suspect: their height, clothing, and behavior. Based on past cases, you estimate the likelihood of each clue belonging to a specific criminal profile. By combining these probabilities, you determine the most probable suspect. This is exactly how Naïve Bayes works—it makes predictions based on probabilities!_

---

### **How Does Naïve Bayes Work?**

Naïve Bayes is a probabilistic classifier based on **Bayes' Theorem**, which states:

> $P(A | B) = \frac{P(B | A) \times P(A)}{P(B)}$

Where:

- $P(A∣B)P(A | B)P(A∣B)$ = Probability of A occurring given B (posterior probability).
- $P(B∣A)P(B | A)P(B∣A)$ = Probability of B occurring given A (likelihood).
- $P(A)P(A)P(A)$ = Probability of A occurring (prior probability).
- $P(B)P(B)P(B)$ = Probability of B occurring (evidence).

**"Naïve"** comes from the assumption that all features are **independent**, which simplifies calculations.

---

### **Story: Classifying Emails as Spam or Not Spam**

> _Imagine you're an email filtering system. Every time an email arrives, you need to decide whether it's spam or not. You analyze the words in the email and assign probabilities based on past emails that were labeled as spam or not spam. If the email contains words like "lottery", "win", or "free money", the probability of it being spam is high!_

---

### **Basic Python Example**

Let's implement a simple **Naïve Bayes classifier** to classify emails as **spam** or **not spam** based on their word content.

#### **Step 1: Import Required Libraries**


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
```

#### **Step 2: Define Training Data**

We create a small dataset where each email is labeled as **spam (1)** or **not spam (0)**.
```python
# List of emails
emails = [
    "Win a lottery now",          # Spam
    "Win free money now",         # Spam
    "Meeting at office today",    # Not Spam
    "Project deadline is tomorrow",# Not Spam
    "Lottery winner announced",   # Spam
    "Money transfer received"     # Not Spam
]

# Corresponding labels: 1 = Spam, 0 = Not Spam
labels = [1, 1, 0, 0, 1, 0]

```



#### **Step 3: Convert Text into Numerical Data**

Naïve Bayes cannot directly process text, so we convert words into a numerical format using **CountVectorizer**.

```python
vectorizer = CountVectorizer() X = vectorizer.fit_transform(emails)
```

``

#### **Step 4: Train the Naïve Bayes Classifier**

```python
`classifier = MultinomialNB() classifier.fit(X, labels)`
```



#### **Step 5: Test on New Data**

Let's classify a new email: **"Win a free lottery ticket now"**.

```python
new_email = ["Win a free lottery ticket now"] X_test = vectorizer.transform(new_email)  prediction = classifier.predict(X_test) print("Spam" if prediction[0] == 1 else "Not Spam")
```


### **Output:**

```python
Spam
```
Since the words **"Win", "lottery", and "free"** were previously associated with spam, the classifier predicts this email as spam.

---

### **Why Use Naïve Bayes?**

✅ **Fast and efficient** – Works well with large datasets.  
✅ **Great for text classification** – Used in spam detection, sentiment analysis, and medical diagnosis.  
✅ **Requires little training data** – Unlike deep learning, which requires tons of labeled data.

🚀 **Want to go further?** Try implementing Naïve Bayes for **movie reviews (positive/negative sentiment)** or **medical diagnoses (disease detection)**!