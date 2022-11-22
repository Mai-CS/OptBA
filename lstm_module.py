import numpy as np
from tensorflow import keras
import tensorflow as tf

import os
import sys

my_lib_path = os.path.abspath("/home/mai.kassem/Documents/ML701-AilmentClassification/")
sys.path.append(my_lib_path)

import preprocessing_module

(
    pad_X_train,
    pad_X_val,
    pad_X_test,
    y_train,
    y_val,
    y_test,
    num_unique_words,
    max_length,
) = preprocessing_module.get_data()


def print_data():
    print(pad_X_train.shape)
    print(y_train.shape)
    print(pad_X_val.shape)
    print(y_val.shape)
    print(pad_X_test.shape)
    print(y_test.shape)
    print(num_unique_words)
    print(max_length)


from tensorflow.keras import layers

# LSTM model
def LSTM(
    embedding_dim=32,
    num_units=64,
    num_classes=25,
    num_epochs=20,
    batch_size=32,
    verbosity=0,
    loss_function="sparse_categorical_crossentropy",
    optimizer="adam",
):

    # Model configuration
    # embedding_dim = 32
    # num_units = 64
    # num_classes = 25
    # num_epochs = 20
    # batch_size = 10
    # verbosity = 0
    # loss_function = 'sparse_categorical_crossentropy'
    # optimizer = 'adam'

    # Create model
    model = keras.models.Sequential()
    model.add(
        layers.Embedding(num_unique_words, embedding_dim, input_length=max_length)
    )
    model.add(
        layers.LSTM(
            num_units,
            dropout=0.2,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0,
            unroll=False,
            use_bias=True,
        )
    )
    model.add(layers.Dense(num_classes, activation="softmax"))
    # Compile model
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
    # Fit the model
    model.fit(
        pad_X_train,
        y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=verbosity,
        validation_data=(pad_X_val, y_val),
    )

    return model


def evaluate_model(model):
    # Evaluate the model on test data
    scores = model.evaluate(pad_X_test, y_test, verbose=0)

    return scores[1]
