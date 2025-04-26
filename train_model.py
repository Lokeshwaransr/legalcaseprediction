import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load Dataset
df = pd.read_csv('land_cases_dataset_1000.csv')

# 2. Remove leading/trailing spaces from column names (if any)
df.columns = df.columns.str.strip()

# 3. Print column names to verify and check for the correct column
print("Columns in dataset:", df.columns)

# 4. Check if 'text' column exists (the actual column for case text)
if 'text' not in df.columns:
    print("Error: 'text' column not found!")
    exit()

# 5. Clean Text Function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 6. Apply Cleaning
df['clean_text'] = df['text'].apply(clean_text)

# 7. Feature and Label
X = df['clean_text']
y = df['problem_type']  # Assuming the problem type is the label for prediction

# 8. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 10. Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 11. Evaluate
score = model.score(X_test_vec, y_test)
print(f"Model Accuracy: {score:.2f}")

# 12. Save Model and Vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and vectorizer saved successfully.")
