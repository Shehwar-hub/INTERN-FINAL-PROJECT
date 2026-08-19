import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("data/jobs_with_skills.csv")


def normalize_department(value):
    value = str(value).strip().lower()

    value = re.sub(
        r"\s+jobs?$",
        "",
        value,
        flags=re.IGNORECASE
    )

    return value.strip()


def map_category(department):
    department = normalize_department(department)

    if department in {
        "it",
        "computer software",
        "web & e-commerce",
        "telecom"
    }:
        return "Technology"

    if department in {
        "sales",
        "marketing",
        "real estate",
        "insurance",
        "management"
    }:
        return "Business & Sales"

    if department == "customer service":
        return "Customer Service"

    if department in {
        "admin",
        "hr & recruiting"
    }:
        return "Administration & HR"

    if department in {
        "finance",
        "accounting"
    }:
        return "Finance"

    if department in {
        "engineering",
        "construction"
    }:
        return "Engineering"

    if department == "education":
        return "Education"

    if department == "healthcare":
        return "Healthcare"

    if department == "production":
        return "Production & Manufacturing"

    return "Other"


df["Target Category"] = df["Department"].apply(
    map_category
)

df["Job Name"] = df["Job Name"].fillna("").astype(str)
df["JD"] = df["JD"].fillna("").astype(str)

df["Model Text"] = (
    df["Job Name"] + " " + df["JD"]
)


X = df["Model Text"]
y = df["Target Category"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)


model.fit(
    X_train_tfidf,
    y_train
)


predictions = model.predict(
    X_test_tfidf
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n" + "=" * 60)
print("JOBLENS BROAD CATEGORY ML MODEL")
print("=" * 60)

print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)

print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


joblib.dump(
    model,
    "models/job_category_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/job_category_vectorizer.pkl"
)


print("\nSaved files:")

print("→ models/job_category_model.pkl")
print("→ models/job_category_vectorizer.pkl")