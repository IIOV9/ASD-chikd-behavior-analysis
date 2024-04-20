# plot_accuracy.py

import random
import json
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Load intents data
data_file = open(".\\intents.json", encoding='utf-8').read()
intents = json.loads(data_file)

# Initialize lists
words = []
classes = []
documents = []

# Tokenization and preprocessing
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize and lemmatize
        w = nltk.word_tokenize(pattern)
        w = [lemmatizer.lemmatize(word.lower()) for word in w]
        words.extend(w)
        documents.append((" ".join(w), intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Shuffle documents
random.shuffle(documents)

# Extract features and labels
X = [doc[0] for doc in documents]
y = [doc[1] for doc in documents]

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Initialize an empty list to store accuracy values for different test sizes
accuracy_values = []

# Test different sizes
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

for size in test_sizes:
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

    # Load the saved model
    with open("chatbot_model.pkl", "rb") as f:
        model, _, _ = pickle.load(f)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    accuracy_values.append(accuracy)

# Plotting the accuracy values
plt.plot(test_sizes, accuracy_values, marker='o')
plt.title('Accuracy vs. Test Size')
plt.xlabel('Test Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
