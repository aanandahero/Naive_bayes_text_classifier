from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Optional: Suppress warning for undefined metrics
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Step 1: Load real news data from sklearn
categories = ['sci.space', 'talk.politics.misc', 'rec.sport.baseball', 'comp.sys.mac.hardware']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.3, random_state=42, stratify=newsgroups.target
)

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 5: Evaluation
y_pred = model.predict(X_test_vec)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=categories))

# Step 6: Predict new example
def predict_news(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return categories[prediction[0]]

sample_text = "NASA is launching a new satellite to explore deep space."
print("\nPredicted Category:", predict_news(sample_text))
