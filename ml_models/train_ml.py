def train_ml_model():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    # Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import MultinomialNB

    from preprocessing.clean_text import clean_text
    from preprocessing.emoji_handler import convert_emojis
    from features.tfidf_vectorizer import get_vectorizer

    print("\n📥 Loading dataset...")
    df = pd.read_csv("data/raw/Algerian Review.csv")

    print("🧹 Preprocessing...")
    df['comment'] = df['comment'].astype(str)
    df['comment'] = df['comment'].apply(convert_emojis)
    df['comment'] = df['comment'].apply(clean_text)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['comment'], df['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']  # مهم جدا بسبب imbalance
    )

    print("🔢 Vectorization (TF-IDF)...")
    vectorizer = get_vectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Models dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "SVM (SVC)": SVC(kernel='linear'),
        "Decision Tree": DecisionTreeClassifier(max_depth=20),
        "Naive Bayes": MultinomialNB()
    }

    results = {}

    print("\n🚀 Training Models...\n")

    for name, model in models.items():
        print(f"🔹 Training: {name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"\n📊 Results for {name}:")
        print(classification_report(y_test, y_pred))
        print("=" * 50)

    # Best model
    best_model = max(results, key=results.get)

    print("\n🏆 BEST MODEL:")
    print(f"{best_model} with accuracy = {results[best_model]:.4f}")

    return results