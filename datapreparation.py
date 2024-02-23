import pickle

from keras.utils import pad_sequences
from keras_preprocessing.text import Tokenizer
import tensorflow as tf
import config as conf
import numpy as np


def tokenize_text(text: str) -> np.ndarray:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        seq = tokenizer.texts_to_sequences([text])

        padded = pad_sequences(seq, maxlen=15, padding="pre")

        return np.array(padded)


def get_word(index: int) -> str:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer: Tokenizer = pickle.load(handle)

        return tokenizer.index_word[index]


def get_tokenizer(text: (list, str)) -> Tokenizer:
    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(text)

    total_words = len(tokenizer.word_index) + 1
    print(f"[INFO] - The size of vocabulary is {total_words}")

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer


def preprocess(text: list[str], tokenizer: Tokenizer):
    input_seq = []
    for line in text:
        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):
            input_seq.append(token_list[:i + 1])

    max_len = max([len(x) for x in input_seq])

    input_seq = pad_sequences(input_seq, maxlen=max_len, padding="pre")

    x, y = input_seq[:, :-1], input_seq[:, -1]

    y = tf.keras.utils.to_categorical(y, num_classes=conf.VOCAB_SIZE)

    return x, y


def get_corpus(file_name: str = "irish-lyrics-eof.txt"):
    with open(file_name) as file:
        corpus = file.read()
        corpus = corpus.lower().split("\n")
        return corpus


def get_train_data():
    text = get_corpus()
    tokenizer = get_tokenizer(text)
    return preprocess(text, tokenizer)


if __name__ == '__main__':
    x, y = get_train_data()
    ...
