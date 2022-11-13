#!/usr/bin/env python
# coding: utf-8

# # Project Idea
# In the recent past, the expansion of the COVID-19 pandemic has reshaped the world radically. Hospitals and medical centers have become a fertile ground for the spread of this virus, where patients are in close contact with someone with COVID-19. Social distancing plays a pivotal role in eliminating the spread of this virus. Hence, a new term appeared, which is telemedicine. Telemedicine is consulting patients by physicians remotely via vast communication technologies. However, the doctors' productivity may decrease due to the intense effort required to balance between in-patients and out-patients. Also, most people try to diagnose themselves by expressing their symptoms in the search engine. Then, they start reading from random unauthorized websites on the internet. On the contrary, this is not safe at all and may lead to the misclassification of the ailment.
# 
# This project aims to speed up the diagnosis process accurately using Natural Language Processing (NLP) models. The dataset that will be used contains more than 6000 text records of variant symptoms along with the type of the ailment. The first step in the proposed work is to perform text preprocessing techniques such as lemmatization, stop words removal and generating word embeddings. Then, deep neural networks will take word embeddings as inputs to predict the output (i.e., the ailment). However, deep learning methods suffer from the risk of getting stuck in local optima. This is because the values of weights are initialized randomly. Not only the weights but also their parameters. The Bees Algorithm (BA) is one of the swarm intelligence algorithms. It is a population-based algorithm as well as it mimics the behavior of the bees in foraging in nature. In the proposed work, the Bees algorithm is used along with deep learning models to enhance the process of hyper-parameter tuning so that the overall performance of classifying unstructured medical text can be improved.

# # Imports

# In[1]:


from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf


# # Load dataset
# https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent

# In[2]:


# ailments_data = pd.read_csv('ailments.csv', encoding="ISO-8859-1")
ailments_data = pd.read_csv('/home/mai.kassem/Documents/ML701-AilmentClassification/aug_ailments_data.csv', encoding="ISO-8859-1")

print(ailments_data.shape)

ailments_data = ailments_data.drop_duplicates()
print(ailments_data.shape)

print("Missing values: ", ailments_data.isnull().sum())

ailments_data.info()

ailments_data.head()


# # Data Preprocessing

# In[3]:


from sklearn import preprocessing
import numpy
# Convert labels to numbers
le = preprocessing.LabelEncoder()
le.fit(ailments_data['prompt'])

le.classes_


y = le.transform(ailments_data['prompt'])
# print(le.inverse_transform(y))


# compare before and after
print(ailments_data['prompt'].value_counts())
unique, counts = numpy.unique(y, return_counts=True)
afterEncoder = dict(zip(unique, counts))
dict(sorted(afterEncoder.items(), key=lambda item: item[1], reverse=True))


# In[4]:


import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
get_ipython().run_line_magic('pip', 'install nltk')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def text_transform(message):

    # (a) change the message to lowercase
    message = message.lower()

    # (b) tokenize the message,
    # i.e. if input = 'i am a student.'
    # then, output  = ['i', 'am', 'a', 'student', '.']
    message = nltk.word_tokenize(message)

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
        if word not in stopwords.words('english') and word not in string.punctuation:
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


ailments_data
# apply the pre-processing steps via text_transform() function on text data
ailments_data['transformed_phrase'] = ailments_data['phrase'].apply(
    text_transform)

X = ailments_data.transformed_phrase
X.tail()


# In[5]:


# count unique words

def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


counter = counter_word(ailments_data['transformed_phrase'])


num_unique_words = len(counter)
print("Total number of unique words : " + str(num_unique_words))

# print(counter)

counted_df = pd.DataFrame(counter.items(), columns=['word', 'count']).sort_values(
    'count', ascending=False).reset_index(drop=True)  # create new df from counter

plt.figure(figsize=(12, 4))
# plot only the top 10 by slicing the df
sns.barplot(data=counted_df[:10], x='word', y='count', alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Word', fontsize=12)
plt.xticks(rotation=90)
plt.show()


# In[6]:


from tensorflow.keras.preprocessing.text import text_to_word_sequence


def get_max_input_length(docs):
    max_input_length = 1
    for document in docs:
        words = text_to_word_sequence(document)
        document_length = len(words)
        if document_length > max_input_length:
            max_input_length = document_length

    return max_input_length


# # Training & Evaluating

# In[7]:


from tensorflow.keras.preprocessing.text import Tokenizer

# Vectorize a text corpus by turning each text into a sequence of integers

tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(X)

# Each word has a unique index

word_index = tokenizer.word_index

word_index


# In[8]:


# Max number of words in a sequence
max_length = get_max_input_length(X)
print("max_length: ", max_length)

train_sequences = tokenizer.texts_to_sequences(X)


# In[9]:


# Pad the sequences to have the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

X = pad_sequences(train_sequences, maxlen=max_length,
                  padding='post', truncating='post')


# In[10]:


# LSTM model
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold


def LSTM(num_folds=10, embedding_dim=32, num_units=64, num_classes=25, num_epochs=20, batch_size=10,
         verbosity=0, loss_function='sparse_categorical_crossentropy', optimizer='adam'):

    # Model configuration
    # num_folds = 10
    # embedding_dim = 32
    # num_units = 64
    # num_classes = 25
    # num_epochs = 20
    # batch_size = 10
    # verbosity = 0
    # loss_function = 'sparse_categorical_crossentropy'
    # optimizer = 'adam'

    # define 10-fold cross validation
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=7)
    cvscores = []
    fold_num = 1
    for train, test in kfold.split(X, y):
        # Create model
        model = keras.models.Sequential()
        model.add(layers.Embedding(num_unique_words,
                                   embedding_dim, input_length=max_length))
        model.add(layers.LSTM(num_units, dropout=0.2, recurrent_dropout=0.2))
        model.add(layers.Dense(num_classes, activation="softmax"))
        # Compile model
        model.compile(loss=loss_function,
                      optimizer=optimizer, metrics=['accuracy'])
        # Fit the model
        print("Training fold number: ", fold_num)
        history = model.fit(X[train], y[train], epochs=num_epochs,
                            batch_size=batch_size, verbose=verbosity,
                            validation_data=(X[test], y[test]))

        # Evaluate the model
        scores = model.evaluate(X[test], y[test], verbose=verbosity)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

        # Visualize history
        # Plot history: Loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Validation loss history')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

        fold_num = fold_num + 1

    print("Average score: %.2f%%" % (np.mean(cvscores)))
    return model


# In[11]:


from tensorflow.keras import models
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


def evaluate_model(model):
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []

    predicted_targets = np.array([])
    actual_targets = np.array([])
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    for train, test in kfold.split(X, y):
        scores = model.evaluate(X[test], y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        y_pred = model.predict(X[test]).argmax(axis=-1)
        predicted_targets = np.append(predicted_targets, y_pred)
        actual_targets = np.append(actual_targets, y[test])

        accuracy_list.append(metrics.accuracy_score(y[test], y_pred))
        precision_list.append(metrics.precision_score(
            y[test], y_pred, average='weighted'))
        recall_list.append(metrics.recall_score(
            y[test], y_pred, average='weighted'))
        f1_score_list.append(metrics.f1_score(
            y[test], y_pred, average='weighted'))

    print("Average accuracy= ", np.mean(accuracy_list))
    print("Average precision= ", np.mean(precision_list))
    print("Average recall= ", np.mean(recall_list))
    print("Average f1_score= ", np.mean(f1_score_list))

    return predicted_targets, actual_targets, np.mean(accuracy_list)


# In[12]:


class_names = le.inverse_transform(y)


# In[ ]:




