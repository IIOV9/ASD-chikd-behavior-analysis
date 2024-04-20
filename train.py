import random
import numpy as np
import pickle
import json
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import wordnet

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Load intents data
data_file = open(".\\intents.json", encoding='utf-8').read()
intents = json.loads(data_file)

# Function to augment data
def augment_data(sentence, n=5):
    augmented_sentences = []
    words = nltk.word_tokenize(sentence)
    for _ in range(n):
        new_words = []
        for word in words:
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = random.choice(synonyms).lemmas()[0].name()
                new_words.append(synonym)
            else:
                new_words.append(word)
        augmented_sentences.append(' '.join(new_words))
    return augmented_sentences

# Initialize lists
documents = []

# Tokenization, lemmatization, and data augmentation
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize and lemmatize
        w = nltk.word_tokenize(pattern)
        w = [lemmatizer.lemmatize(word.lower()) for word in w]
        documents.append((" ".join(w), intent["tag"]))
        
        # Augment data
        augmented_patterns = augment_data(pattern, n=5)
        for augmented_pattern in augmented_patterns:
            w_augmented = nltk.word_tokenize(augmented_pattern)
            w_augmented = [lemmatizer.lemmatize(word.lower()) for word in w_augmented]
            documents.append((" ".join(w_augmented), intent["tag"]))

# Shuffle documents
random.shuffle(documents)

# Extract features and labels
X = [doc[0] for doc in documents]
y = [doc[1] for doc in documents]

# Vectorize text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the training set
train_accuracy = model.score(X_train, y_train)
print("Training set accuracy:", train_accuracy)

# Predict labels for training set
y_train_pred = model.predict(X_train)

# Calculate accuracy for each training data point
train_data_accuracies = [1 if pred == true else 0 for pred, true in zip(y_train_pred, y_train)]
print("Accuracy for each training data point:", train_data_accuracies)

# Plot the accuracy for each training data point
plt.figure(figsize=(10, 6))
plt.plot(train_data_accuracies, marker='o', linestyle='')
plt.xlabel('Training Data Point Index')
plt.ylabel('Accuracy')
plt.title('Accuracy of Each Training Data Point')
plt.ylim(0, 1.1)
plt.show()

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Plot the accuracy
plt.bar(['Accuracy'], [accuracy], color='skyblue')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.ylim(0, 1)
plt.show()

# Save the model and preprocessing objects
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, label_encoder), f)

print("Model saved successfully.")
