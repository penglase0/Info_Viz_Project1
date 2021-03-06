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
#df_filtered = pd.read_csv('Phase2Dataset_sentiment_3labels.csv')
#df_filtered = pd.read_csv('Phase2Dataset_v4.csv')

df_filtered = pd.read_csv('phase_2_multilabel.csv')

# rename original adjusted_rating column and adjust the one to be used to start at 0
df_filtered['adjusted_rating1'] = df_filtered['adjusted_rating']
#df_filtered['adjusted_rating'] = df_filtered['adjusted_rating'] + 1

# change adjusted rating so no values are based on positive column
def new_rating(row):
    if (row['rating']<4): # assign 0 to ratings of 0, 1, 2, 3
        return 0
    elif (row['rating']>=4 and row['rating']<8):
        return 1
    else:
        return 2


df_filtered['adjusted_rating'] = df_filtered.apply(new_rating, axis=1)


# BERT Sentiment Analysis #############################################################################################

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



#model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",
#                                                        num_labels=3,
#                                                        output_attentions=False,
#                                                        output_hidden_states=False)

#model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# summary of the BERT model
#model.summary()

import tensorflow as tf
import pandas as pd

# https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671

# split the data into a training dataset and a test dataset, use ShuffleSplit to ensure it's random
split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

for train_index, test_index in split.split(df_filtered):
    train_set = df_filtered.iloc[train_index]
    test_set = df_filtered.iloc[test_index]

# trim the dataset for the only necessary columns, the cleaned survery responses and numeric sentiment (-1, 0, 1)
train = train_set[['ResponseID', 'clean', 'adjusted_rating']].copy()
test = test_set[['ResponseID', 'clean', 'adjusted_rating']].copy()


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

# create InputExample objects
train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, data_column, label_column)
print('Finished creating the Input Examples')

# tokenize the InputExample objects for the training data
train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)
print('Finished creating the training tf dataset')

# tokenize the InputExample objects for the validation data
validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)
print('Finished creating the test tf dataset')

# Adam Optimizer - long run time, not used
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

# SGD Optimizer - 63% Accuracy
# briefly decreased learning rate to 0.00001 (or something smaller like that)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

# Adagrad Optimizer - 23% Accuracy
#model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-0-7 , name='Adagrad'),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])


print('Finished compiling the model')

# previously had a batch size of like 500
model.fit(train_data, epochs=1, validation_data=validation_data)
print('Finished fitting the model with ', 1, ' epochs.')

# save the model
#model.save('sentiment_model')

#for i in range(1, 4):
    #model.fit(train_data, epochs=i,  validation_data=validation_data)
    #print('Finished fitting the model with ', i, ' epochs.')

# Information about Batch & Epochs: https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
# Other Optimizers to Do: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# Pros/Cons of Optimizers: https://www.dlology.com/blog/quick-notes-on-how-to-choose-optimizer-in-keras/

# run model for entire dataset

# get the clean comment column and make it a list
#TF_GPU_ALLOCATOR=cuda_malloc_async
#response_list = test['clean'].head(650).tolist()
#id_list = test['ResponseID'].head(650).tolist()
#print(len(response_list))





# predict full data set 650 rows at a time

response_list = df_filtered['clean'].tolist()
id_list = df_filtered['ResponseID'].tolist()
assigned_rating = df_filtered['adjusted_rating']

# create empty lists to append to to output
all_labels = []


start_index = 0
process_row_length = 500
count_check = 0

if (start_index + process_row_length < len(response_list)):
    while (start_index + process_row_length < len(response_list)):
        end_index = start_index+process_row_length

        print(len(response_list[start_index:end_index]))

        print(count_check)
        count_check +=1
        tf_batch = tokenizer(response_list[start_index:end_index],
                            max_length=128,
                            padding=True,
                            truncation=True,
                            return_tensors='tf')
        tf_outputs = model(tf_batch)
        #print('TF OUTPUTS: \n', tf_outputs)

        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=1)
        print('TF PREDICTIONS: \n', tf_predictions)
        # tf_predictions = tf.nn.sigmoid(tf_outputs[0], axis=1)

        label = tf.argmax(tf_predictions, axis=1)
        print('LABEL: \n', tf_predictions)
        label = label.numpy()
        #for i in range(len(response_list)):
            #print(response_list[i], ": \n", label[i])

        #dict = {'ID': id_list, 'response': response_list, 'prediction': label}
        #prediction_output = pd.DataFrame(dict)
        #prediction_output.to_csv('prediction_1k.csv')

        all_labels = np.append(all_labels, label)

        print(count_check)
        start_index += process_row_length
    else:
        end_index = len(response_list)+1

        print(len(response_list[start_index:end_index]))

        print(count_check)
        count_check += 1
        tf_batch = tokenizer(response_list[start_index:end_index],
                             max_length=128,
                             padding=True,
                             truncation=True,
                             return_tensors='tf')
        tf_outputs = model(tf_batch)
        # print('TF OUTPUTS: \n', tf_outputs)

        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=1)
        print('TF PREDICTIONS: \n', tf_predictions)
        # tf_predictions = tf.nn.sigmoid(tf_outputs[0], axis=1)

        label = tf.argmax(tf_predictions, axis=1)
        print('LABEL: \n', tf_predictions)
        label = label.numpy()
        # for i in range(len(response_list)):
        # print(response_list[i], ": \n", label[i])

        # dict = {'ID': id_list, 'response': response_list, 'prediction': label}
        # prediction_output = pd.DataFrame(dict)
        # prediction_output.to_csv('prediction_1k.csv')

        all_labels = np.append(all_labels, label)

        print(count_check)
        start_index += process_row_length


#print('id: ', len(id_list))
#print('response: ', len(response_list))
#print('predict: ', len(all_labels))

#dict = {'ID': id_list, 'response': response_list, 'prediction': all_labels}
#rediction_output = pd.DataFrame(dict)

df_filtered['prediction'] = all_labels.tolist()
df_filtered.to_csv('phase_2_multilabel_final.csv')









#df_filtered_test = pd.read_csv('Phase2Dataset_v4final.csv')
#response_list = df_filtered_test['clean'].to_list()
#print(type(response_list))
#print(response_list.head())

#response_list = ['This was an awesome movie. I watch it twice my time watching this beautiful movie if I have known it was this good',
#                  'One of the worst movies of all time. I cannot believe I wasted two hours of my life for this movie',
#                 'super amazing',
#                 'terrible service, dont recommend this product',
#                 'this is only ok']

#response_list = ['This was an awesome movie. I watch it twice my time watching this beautiful movie if I have known it was this good']


# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
#session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))



#tf_batch = tokenizer(response_list, max_length=128, padding=True, truncation=True, return_tensors='tf')
#probs = model.predict(tf_batch)[0]
#print(probs)
#tf_outputs = model(tf_batch)
#print('TF OUTPUTS: \n', tf_outputs)

#tf_predictions = tf.nn.softmax(tf_outputs[0], axis=1)
#print('TF PREDICTIONS: \n', tf_predictions)
#tf_predictions = tf.nn.sigmoid(tf_outputs[0], axis=1)

#labels = ['negative', 'neutral', 'positive']
#label = tf.argmax(tf_predictions, axis=1)
#print('LABEL: \n', tf_predictions)
#label = label.numpy()
#for i in range(len(response_list)):
#  print(response_list[i], ": \n", label[i])

#dict = {'ID': id_list, 'response': response_list, 'prediction': label}
#prediction_output = pd.DataFrame(dict)
#prediction_output.to_csv('prediction_1k.csv')

