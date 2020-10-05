from models.model import MODEL
import helper
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    input_size = 10000
    embedding_size = 24
    output_size = 5
    learning_rate = 0.01
    oov_token = '<OOV>'
    loss = 'sparse_categorical_crossentropy'
    optimizer = Adam(learning_rate=learning_rate)
    epochs = 1
    train_val_split = 0.2

    sentences, sentiments = helper.get_data('data/train.tsv')
    sentences = helper.remove_stopwords(sentences, 'data/stopwords')

    max_length = len(max(sentences, key=len))

    tokenizer = helper.get_tokenizer(input_size, oov_token, sentences)

    padded_sentences = helper.convert_to_sequences(tokenizer, sentences, max_length)

    train_padded_sentences, validation_padded_sentences, train_sentiments, validation_sentiments = \
        train_test_split(
            padded_sentences, sentiments, test_size=train_val_split, random_state=42
        )

    train_padded_sentences = np.array(train_padded_sentences)
    train_sentiments = np.array(train_sentiments)
    validation_padded_sentences = np.array(validation_padded_sentences)
    validation_sentiments = np.array(validation_sentiments)

    layers = [
        tf.keras.layers.Embedding(input_size, embedding_size, input_length=max_length),
        # tf.keras.layers.LSTM(32),

        # tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
        # tf.keras.layers.MaxPooling1D(pool_size=4),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(units=24, activation='relu'),
        tf.keras.layers.Dense(units=output_size, activation='softmax')
    ]

    model = MODEL(input_size, output_size, layers, loss, optimizer, epochs)
    model.__train__(train_padded_sentences, train_sentiments, validation_padded_sentences,
                    validation_sentiments)
    model.__plot_graph__('accuracy')


main()
