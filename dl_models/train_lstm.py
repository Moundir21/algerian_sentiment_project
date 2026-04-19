def train_lstm_model():
    import pandas as pd
    import pickle

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from sklearn.model_selection import train_test_split

    from dl_models.lstm_model import build_lstm
    from preprocessing.clean_text import clean_text

    # Load dataset
    df = pd.read_csv("data/raw/Algerian Review.csv")

    # Clean text
    df['comment'] = df['comment'].astype(str).apply(clean_text)

    # Filter labels
    df = df[df['sentiment'].isin([0, 1, 2])]
    df['sentiment'] = df['sentiment'].astype(int)

    # 🔥 (1) Tokenization محسن
    max_words = 15000   # كان 10000 → زدناه
    max_len = 120       # كان 100 → زدناه

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['comment'])

    sequences = tokenizer.texts_to_sequences(df['comment'])
    X = pad_sequences(sequences, maxlen=max_len, padding='post')

    y = df['sentiment']

    # 🔥 (2) حفظ tokenizer (مهم للتجربة لاحقًا)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build model
    model = build_lstm(vocab_size=max_words, max_len=max_len, num_classes=3)

    # 🔥 (3) EarlyStopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # 🔥 (4) Reduce Learning Rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        min_lr=1e-5
    )

    # 🔥 (5) حفظ أفضل Model
    checkpoint = ModelCheckpoint(
        "best_lstm_model.h5",
        monitor='val_accuracy',
        save_best_only=True
    )

    # 🔥 (6) Training محسن
    history = model.fit(
        X_train, y_train,
        epochs=20,             # زدنا epochs
        batch_size=32,         # قللناها لتحسين التعلم
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n✅ LSTM Accuracy: {acc:.4f}")