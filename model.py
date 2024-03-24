import numpy as np
from tensorflow import keras
import config as conf
from datapreparation import get_train_data, tokenize_text, get_word
import matplotlib.pyplot as plt


def create_model():
    model = keras.models.Sequential([
        keras.layers.Embedding(conf.VOCAB_SIZE, 128, input_length=15),
        keras.layers.Bidirectional(keras.layers.GRU(64)),
        keras.layers.Dense(4096, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(conf.VOCAB_SIZE, activation="softmax")
    ])

    model.compile("adam", "categorical_crossentropy", ["accuracy"])

    return model


def train_model(model: keras.Model):
    x, y = get_train_data()

    history = model.fit(x, y, epochs=20, validation_split=0.15)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.show()

    model.save("trained_model")


def test_model(text: str, pred_size: int = 10):
    model = keras.models.load_model("trained_model")

    for _ in range(pred_size):
        tokenized_text = tokenize_text(text)

        pp = model.predict(tokenized_text)
        prediction = np.argmax(pp, axis=-1)[0]

        if prediction != 0:
            word = get_word(prediction)
            text += " " + word

    return text


if __name__ == '__main__':
    train_model(create_model())
