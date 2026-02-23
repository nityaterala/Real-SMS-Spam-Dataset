# Real SMS Spam Detector using NLP + Machine Learning

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# STEP 1: Load Real Dataset
# -----------------------------
df = pd.read_csv("SMSSpamCollection", sep='\t', names=["label", "message"])

print(df.head())

# -----------------------------
# STEP 2: Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["message"] = df["message"].apply(clean_text)

# -----------------------------
# STEP 3: Features & Labels
# -----------------------------
X = df["message"]
y = df["label"]

# -----------------------------
# STEP 4: Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 5: TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# STEP 6: Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# STEP 7: Evaluate Model
# -----------------------------
predictions = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nReport:\n", classification_report(y_test, predictions))

# -----------------------------
# STEP 8: Test Custom Message
# -----------------------------
new_msg = ["Congratulations! You won a free iPhone. Click now!"]
new_msg_vec = vectorizer.transform(new_msg)

result = model.predict(new_msg_vec)
print("\nPrediction:", result[0])
