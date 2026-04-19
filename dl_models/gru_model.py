from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Input

def build_gru(vocab_size, max_len, num_classes=3):
    """
    Build a simple GRU model for multi-class sentiment classification.
    """
    model = Sequential([
        Input(shape=(max_len,)),                       # Input layer
        Embedding(input_dim=vocab_size, output_dim=128),

        GRU(128, return_sequences=False),              # GRU instead of LSTM

        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')       # Output layer
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model