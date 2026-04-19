def train_gru_model():
    import pandas as pd
    import pickle

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split

    from dl_models.gru_model import build_gru
    from preprocessing.clean_text import clean_text

    # Load dataset
    df = pd.read_csv("data/raw/Algerian Review.csv")

    # Clean text
    df['comment'] = df['comment'].astype(str).apply(clean_text)

    # Filter valid labels
    df = df[df['sentiment'].isin([0, 1, 2])]
    df['sentiment'] = df['sentiment'].astype(int)

    # Tokenization
    max_words = 10000
    max_len = 100

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df['comment'])

    sequences = tokenizer.texts_to_sequences(df['comment'])
    X = pad_sequences(sequences, maxlen=max_len)

    y = df['sentiment']

    # Save tokenizer (مهم)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build GRU
    model = build_gru(vocab_size=max_words, max_len=max_len, num_classes=3)

    # Train
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n✅ GRU Accuracy: {acc:.4f}")

    # Save model
    model.save("gru_model.h5")