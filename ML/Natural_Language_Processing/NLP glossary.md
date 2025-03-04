- **Tokenization**: Breaking down text into individual units, such as words or sentences.
    
    _Example_:

```python
from nltk.tokenize import word_tokenize
text = "Hello, world!"
tokens = word_tokenize(text) print(tokens)  
# Output: ['Hello', ',', 'world', '!']
```
    
    
- **Stop Words**: Commonly used words (e.g., 'and', 'the') that are often removed from text to focus on meaningful content.
    
    _Example_:
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
text = "This is a sample sentence."
stop_words = set(stopwords.words('english'))
words = word_tokenize(text)
filtered_words = [word for word in words if word.lower() not in stop_words] print(filtered_words) 
# Output: ['This', 'sample', 'sentence', '.']
```
    
    
- **Stemming**: Reducing words to their root form.
    
    _Example_:
```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
words = ["running", "ran", "runs"]
stems = [stemmer.stem(word) for word in words] print(stems)
# Output: ['run', 'ran', 'run']
```
    
- **Lemmatization**: Converting words to their base or dictionary form.
    
    _Example_:
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
words = ["running", "ran", "runs"]
lemmas = [lemmatizer.lemmatize(word, pos='v') for word in words] print(lemmas)  # Output: ['run', 'run', 'run']
```
    
- **Part-of-Speech Tagging (POS Tagging)**: Identifying the grammatical role of words in a sentence.
    
    _Example_:
```python
from nltk import pos_tag
from nltk.tokenize import word_tokenize
text = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
print(pos_tags)
# Output: [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'),
#          ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]

```

    
- **Named Entity Recognition (NER)**: Identifying and classifying entities in text into predefined categories like names, organizations, locations, etc.
    
    _Example_:
```python
import spacy
nlp = spacy.load('en_core_web_sm')
text = "Apple is looking at buying U.K. startup for $1 billion."
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
# Output:
# Apple ORG
# U.K. GPE
# $1 billion MONEY

```
   
- **Bag of Words (BoW)**: Representing text by the frequency of words, disregarding grammar and word order.
    
    _Example_:
```python
from sklearn.feature_extraction.text import CountVectorizer
texts = ["I love NLP.", "NLP is great!"]
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(texts)
print(bow.toarray())
# Output: [[0 1 1 1]
#          [1 0 1 1]]
print(vectorizer.get_feature_names_out())
# Output: ['great', 'love', 'nlp', 'is']

```

    
- **Term Frequency-Inverse Document Frequency (TF-IDF)**: A statistical measure to evaluate the importance of a word in a document relative to a collection of documents.
    
    _Example_:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
texts = ["I love NLP.", "NLP is great!"]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(texts)
print(tfidf.toarray())
# Output: [[0.         0.70710678 0.70710678 0.        ]
#          [0.70710678 0.         0.         0.70710678]]
print(vectorizer.get_feature_names_out())
# Output: ['great', 'love', 'nlp', 'is']

```
    
    
- **Word Embeddings**: Representing words in continuous vector space, capturing semantic relationships.
    
    _Example_:
```python
from gensim.models import Word2Vec
sentences = [["I", "love", "NLP"], ["I", "love", "coding"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
vector = model.wv['NLP']
print(vector)  # Output: Vector representation of the word 'NLP'

```

    
- **[[Padding]]**: Ensuring all sequences in a dataset have the same length by adding extra tokens.
    
    _Example_:
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
sequences = [[1, 2, 3], [4, 5], [6]]
padded = pad_sequences(sequences, maxlen=4, padding='post')
print(padded)
# Output:
# [[1 2 3 0]
#  [4 5 0 0]
#  [6 0 0 0]]

```
    
