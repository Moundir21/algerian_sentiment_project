def train_gru_model():
    import pandas as pd
    import pickle

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from sklearn.model_selection import train_test_split

    from dl_models.gru_model import build_gru
    from preprocessing.clean_text import clean_text

    # 📂 Load dataset
    df = pd.read_csv("data/raw/Algerian Review.csv")

    # 🧹 Clean text
    df['comment'] = df['comment'].astype(str).apply(clean_text)

    # 🎯 Filter labels
    df = df[df['sentiment'].isin([0, 1, 2])]
    df['sentiment'] = df['sentiment'].astype(int)

    # 🔥 Tokenization improved
    max_words = 15000   # كان 10000 → زيادة vocabulary
    max_len = 120       # كان 100 → جمل أطول = فهم أفضل

    tokenizer = Tokenizer(
        num_words=max_words,
        oov_token="<OOV>"   # 🔥 كلمات غير معروفة
    )
    tokenizer.fit_on_texts(df['comment'])

    sequences = tokenizer.texts_to_sequences(df['comment'])
    X = pad_sequences(sequences, maxlen=max_len, padding='post')

    y = df['sentiment']

    # 💾 حفظ tokenizer
    with open("gru_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # 🔀 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🧠 Build model
    model = build_gru(vocab_size=max_words, max_len=max_len, num_classes=3)

    # 🔥 callbacks (سر تحسين الدقة)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        min_lr=1e-5
    )

    checkpoint = ModelCheckpoint(
        "best_gru_model.h5",
        monitor='val_accuracy',
        save_best_only=True
    )

    # 🚀 training improved
    history = model.fit(
        X_train, y_train,
        epochs=20,           # كان 10 → زيادة learning
        batch_size=32,       # كان 64 → تحسين generalization
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    # 📊 evaluation
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n✅ Improved GRU Accuracy: {acc:.4f}")

    # 💾 save final model
    model.save("gru_model.h5")