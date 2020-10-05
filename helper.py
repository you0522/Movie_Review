import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_stopwords(filename):
    with open(filename, 'r') as f:
        stopwords = f.read().splitlines()
    return stopwords


def get_data(filename):
    sentiments_index = 3
    sentences_index = 2
    with open(filename, 'r') as f:
        data = csv.reader(f, delimiter='\t')
        sentences = []
        sentiments = []
        next(data)
        for item in data:
            sentiments.append(int(item[sentiments_index]))
            sentences.append(item[sentences_index])
    return sentences, sentiments


def remove_stopwords(sentences, stopwords_filename):
    stopwords = get_stopwords(stopwords_filename)
    modified = []
    for sentence in sentences:
        for stop_word in stopwords:
            sentence = sentence.replace(" " + stop_word + " ", " ")
        modified.append(sentence)
    return modified


def get_tokenizer(num_words, oov_token, sentences):
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(sentences)
    return tokenizer


def convert_to_sequences(tokenizer, sentences, max_length):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length,
                                     truncating='post', padding='post')
    return padded_sequences
