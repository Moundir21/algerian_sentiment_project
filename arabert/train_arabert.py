def train_arabert_model():
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    import pandas as pd

    model_name = "aubmindlab/bert-base-arabertv2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    df = pd.read_csv("data/raw/Algerian Review.csv")

    texts = list(df['comment'])
    labels = list(df['sentiment'])

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        labels
    )).batch(16)

    model.compile(
        optimizer=Adam(learning_rate=2e-5),
        loss=model.compute_loss,
        metrics=['accuracy']
    )

    model.fit(dataset, epochs=3)