# runarabert.py

import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def train_arabert_model():

    # 🔥 1. تحميل اسم النموذج
    model_name = "aubmindlab/bert-base-arabertv2"

    # 🔥 2. تحميل tokenizer و model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    # 🔥 3. تحميل البيانات
    df = pd.read_csv("data/raw/Algerian Review.csv")

    # حذف القيم الفارغة
    df = df.dropna()

    texts = list(df['comment'])
    labels = list(df['sentiment'])

    # 🔥 4. تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # 🔥 5. Tokenization
    train_encodings = tokenizer(
        X_train,
        truncation=True,
        padding=True,
        max_length=128
    )

    test_encodings = tokenizer(
        X_test,
        truncation=True,
        padding=True,
        max_length=128
    )

    # 🔥 6. تحويل إلى Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train
    )).shuffle(1000).batch(16)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test
    )).batch(16)

    # 🔥 7. Compile
    model.compile(
        optimizer=Adam(learning_rate=2e-5),
        loss=model.compute_loss,
        metrics=['accuracy']
    )

    # 🔥 8. Training
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=3
    )

    # 🔥 9. حفظ النموذج
    model.save_pretrained("arabert_model")
    tokenizer.save_pretrained("arabert_model")

    print("✅ Training finished and model saved!")


if __name__ == "__main__":
    train_arabert_model()