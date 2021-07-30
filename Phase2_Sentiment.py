# Authors: Courtney Datin and Oliver Penglase
# Course: DS5500 - Information Visualization in Data Science
# Project Phase 2 - Classification of Student Response Surveys

# import necessary packages
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import ShuffleSplit

# enable GPU to train model
import tensorflow as tf
#sess = tf.Session()
from tensorflow import keras
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# read csv file with cleaned text and numeric sentiment labels
df_filtered = pd.read_csv('Phase2Dataset_sentiment_3labels.csv')


# BERT Sentiment Analysis #############################################################################################
# https://towardsdatascience.com/lstm-vs-bert-a-step-by-step-guide-for-tweet-sentiment-analysis-ced697948c47

def tokenize_text(text):
    """
    Tokenize the text input using NLTKs word_tokenize()
    :param text: the text field the
    :return: tokenized text response
    """
    return [word for word in word_tokenize(text) if (word.isalpha() == 1)]

tokenized_comments = pd.Series([tokenize_text(row) for row in df_filtered])

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# summary of the BERT model
#model.summary()

import tensorflow as tf
import pandas as pd

# https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671
# actually i think it's this one https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671

# split the data into a training dataset and a test dataset, use ShuffleSplit to ensure it's random
split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

for train_index, test_index in split.split(df_filtered):
    train_set = df_filtered.iloc[train_index]
    test_set = df_filtered.iloc[test_index]

# trim the dataset for the only necessary columns, the cleaned survery responses and numeric sentiment (-1, 0, 1)
train = train_set[['clean', 'adjusted_rating']].copy()
test = test_set[['clean', 'adjusted_rating']].copy()


# Convert the two pandas dataframes into suitable objects for the BERT model by using the InputExample function
# below calls the InputExample function
InputExample(guid=None,
             text_a = "Hello, world",
             text_b = None,
             label = 1)

def convert_data_to_examples(train, test, data_column, label_column):
    """
    Function that accepts the train and test datasets and converts each row into an InputExample object
    :param train: train dataset
    :param test: test dataset
    :param data_column: name of the data column (column of the cleaned survey responses)
    :param label_column: name of the label column (numerical indicator of the labeled sentiment)
    :return: the train and test dataframes converted to InputExample objects
    """
    train_InputExamples = train.apply(lambda x: InputExample(guid=None,
                                                             text_a = x[data_column],
                                                             text_b = None,
                                                             label = x[label_column]), axis = 1)


    validation_InputExamples = test.apply(lambda x: InputExample(guid=None,
                                                             text_a = x[data_column],
                                                             text_b = None,
                                                             label = x[label_column]), axis = 1)

    return train_InputExamples, validation_InputExamples


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    """
    tokenizes the survey responses in the InputExample objects
    :param examples: text example
    :param tokenizer: tokenizer (currently using BertTokenizer bert-based-uncased)
    :param max_length: maximum length of tokens, default is 128
    :return:  dataset for the model
    """

    features = [] # temporary list to hold the InputFeatures
    for row in examples:
        input_dict = tokenizer.encode_plus(
            row.text_a,
            add_special_tokens = True,
            max_length = max_length, # truncates if the len(s) > max_length
            return_token_type_ids = True,
            return_attention_mask = True,
            pad_to_max_length = True,
            truncation = True
        )
        input_ids, token_type_ids, attention_mask = (input_dict['input_ids'],
                                                     input_dict['token_type_ids'],
                                                     input_dict['attention_mask'])
        features.append(
            InputFeatures( input_ids = input_ids,
                           attention_mask = attention_mask,
                           token_type_ids = token_type_ids ,
                           label = row.label)
        )

    def gen():
        for i in features:
            yield (
                {'input_ids':i.input_ids,
                 'attention_mask': i.attention_mask,
                 'token_type_ids': i.token_type_ids,},
                i.label
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({'input_ids': tf.int32, 'attention_mask': tf.int32, 'token_type_ids': tf.int32}, tf.int64),
        ({
            "input_ids": tf.TensorShape([None]),
            "attention_mask": tf.TensorShape([None]),
            "token_type_ids": tf.TensorShape([None]),
        }, tf.TensorShape([]),
        ),
    )


data_column = 'clean'
label_column = 'adjusted_rating'


train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, data_column, label_column)
print('Finished creating the Input Examples')

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)
print('Finished creating the training tf dataset')

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)
print('Finished creating the test tf dataset')


#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

print('Finished compiling the model')

model.fit(train_data, epochs=1, batch_size = 200, validation_data=validation_data)
print('Finished fitting the model')

# Information about Batch & Epochs: https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/


