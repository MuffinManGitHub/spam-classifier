from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

csv_path = Path(__file__).resolve().parent / 'spam.csv'

# The file contains Windows/Latin-1 characters like £, so use latin1.
# The CSV has extra blank columns after the message, so read only the first two fields.
df = pd.read_csv(
    csv_path,
    encoding='latin1',
    dtype=str,
    keep_default_na=False,
    on_bad_lines='skip',
    usecols=[0, 1],
)
df.columns = ['label', 'message']

df = df.dropna(subset=['message']).copy()

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
    ("clf", LinearSVC())
])

# TFIDF model performs the best

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
