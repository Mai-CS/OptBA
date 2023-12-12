from collections import Counter
import pandas as pd

import tensorflow as tf
from tensorflow.python.client import device_lib

print(tf.config.list_physical_devices("GPU"))
print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
print(tf.test.is_built_with_cuda())


print(device_lib.list_local_devices())

# Load dataset
# https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent

root_path = "./data/"
ailments_data = pd.read_csv(
    root_path + "aug_ailments.csv",
    encoding="ISO-8859-1",
)

ailments_data = ailments_data.drop_duplicates()


# Text Preprocessing
from sklearn import preprocessing

# convert labels to numbers
le = preprocessing.LabelEncoder()
le.fit(ailments_data["prompt"])

# print(le.classes_)

y = le.transform(ailments_data["prompt"])
# print(le.inverse_transform(y))

import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


def text_transform(message):
    """Tokenization and lemmatization of text

    Args:
        message (string): input text

    Returns:
        string: text after transformation
    """

    # (a) change the message to lowercase
    message = message.lower()

    # (b) tokenize the message,
    # i.e. if input = 'i am a student.'
    # then, output  = ['i', 'am', 'a', 'student', '.']
    message = word_tokenize(message)

    # (c) remove special characters in the message
    msg_temp = []
    for word in message:
        # only accpet the alpha-numeric words and remove all other cases e.g. special characters
        if word.isalnum():
            msg_temp.append(word)

    message = msg_temp

    # (d) remove stopwords and punctuations
    msg_temp = []
    for word in message:
        if word not in stopwords.words("english") and word not in string.punctuation:
            msg_temp.append(word)

    message = msg_temp

    # (e) lemmatization function
    lemmatizer = WordNetLemmatizer()
    msg_temp = []
    for word in message:
        msg_temp.append(lemmatizer.lemmatize(word))

    # join all words with space and return new message
    new_message = " ".join(msg_temp)

    return new_message


# apply the pre-processing steps via text_transform() function on text data
ailments_data["transformed_phrase"] = ailments_data["phrase"].apply(text_transform)

X = ailments_data.transformed_phrase


# count unique words
def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


counter = counter_word(ailments_data["transformed_phrase"])
num_unique_words = len(counter)

from tensorflow.keras.preprocessing.text import text_to_word_sequence


def get_max_input_length(docs):
    max_input_length = 1
    for document in docs:
        words = text_to_word_sequence(document)
        document_length = len(words)
        if document_length > max_input_length:
            max_input_length = document_length

    return max_input_length


from tensorflow.keras.preprocessing.text import Tokenizer

# vectorize a text corpus by turning each text into a sequence of integers
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(X)

# each word has a unique index
word_index = tokenizer.word_index

# max number of words in a sequence
max_length = get_max_input_length(X)
print("max_length: ", max_length)


def get_data():
    # split data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    train_sequences = tokenizer.texts_to_sequences(X_train)
    val_sequences = tokenizer.texts_to_sequences(X_val)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    # pad the sequences to have the same length
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    pad_X_train = pad_sequences(
        train_sequences, maxlen=max_length, padding="post", truncating="post"
    )
    pad_X_val = pad_sequences(
        val_sequences, maxlen=max_length, padding="post", truncating="post"
    )
    pad_X_test = pad_sequences(
        test_sequences, maxlen=max_length, padding="post", truncating="post"
    )

    return (
        pad_X_train,
        pad_X_val,
        pad_X_test,
        y_train,
        y_val,
        y_test,
        num_unique_words,
        max_length,
    )
