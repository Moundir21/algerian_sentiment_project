#gru_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Input, Dropout, Bidirectional

def build_gru(vocab_size, max_len, num_classes=3):
    """
    Improved GRU model for higher accuracy (~90%)
    """

    model = Sequential([
        Input(shape=(max_len,)),

        # 🔥 (1) Embedding أقوى
        Embedding(
            input_dim=vocab_size,
            output_dim=200   # كان 128 → زيادة تمثيل الكلمات
        ),

        # 🔥 (2) Bidirectional GRU
        # يفهم الجملة من الاتجاهين → أفضل بكثير في sentiment
        Bidirectional(GRU(128, return_sequences=False)),

        # 🔥 (3) Dropout لتقليل overfitting
        Dropout(0.5),

        # 🔥 (4) Dense أقوى
        Dense(128, activation='relu'),

        # 🔥 (5) Dropout إضافي
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model