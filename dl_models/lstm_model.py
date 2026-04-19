from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Dropout, Bidirectional

def build_lstm(vocab_size, max_len, num_classes=3):
    """
    Improved LSTM model for better accuracy
    """

    model = Sequential([
        Input(shape=(max_len,)),

        # 🔥 (1) تحسين Embedding
        # زدنا output_dim من 128 → 200 لتمثيل الكلمات بشكل أفضل
        Embedding(input_dim=vocab_size, output_dim=200),

        # 🔥 (2) Bidirectional LSTM
        # يقرأ الجملة من اليمين واليسار → يفهم السياق أفضل
        Bidirectional(LSTM(128, return_sequences=False)),

        # 🔥 (3) Dropout
        # يقلل Overfitting
        Dropout(0.5),

        # 🔥 (4) Dense أقوى
        Dense(128, activation='relu'),

        # 🔥 (5) Dropout ثاني
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    # 🔥 (6) تحسين optimizer
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model