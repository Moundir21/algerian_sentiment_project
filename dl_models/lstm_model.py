from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input

def build_lstm(vocab_size, max_len, num_classes=3):
    """
    Build a simple LSTM model for multi-class sentiment classification.
    """
    model = Sequential([
        Input(shape=(max_len,)),        # Input layer
        Embedding(input_dim=vocab_size, output_dim=128),  # Remove input_length
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')          # Output layer
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model