# Authors: Courtney Datin and Oliver Penglase
# Course: DS5500 - Information Visualization in Data Science
# Project Phase 2 - Classification of Student Response Surveys


# Process raw data to have more comprehensive sentiment labeling ######################################################

# import necessary packages
import pandas as pd
import numpy as np

# read in csv dataset as a dataframe
survey_comments = pd.read_csv('Phase2Dataset.csv')

pos_survey_comments = survey_comments[survey_comments['positive'] == 1].reset_index(drop=True)
unavailable_survey_comments = survey_comments[survey_comments['positive'] != 1].reset_index(drop=True)

# create a list of the conditions for the rating column (negative, neutral, positive)
conditions = [
    (unavailable_survey_comments['rating'] <= 3),
    (unavailable_survey_comments['rating'] > 3) & (unavailable_survey_comments['rating'] <=7),
    (unavailable_survey_comments['rating'] >7)
]

# create a list of possible values for the rating conditions (negative = -1, neutral = 0, positive = 1)
assigned_values = [0, .5, 1]

# create a new column for the values
unavailable_survey_comments['adjusted_rating'] = np.select(conditions, assigned_values)

# write dataframe with adjusted rating column to csv
pos_survey_comments['adjusted_rating'] = pos_survey_comments['positive']
combined_survey = pos_survey_comments.append(unavailable_survey_comments)

conditions_text = [
    (combined_survey['adjusted_rating'] == 0),
    (combined_survey['adjusted_rating'] == .5),
    (combined_survey['adjusted_rating'] == 1)
]

assigned_values_text = ['negative', 'neutral', 'positive']
combined_survey['adjusted_rating_text'] = np.select(conditions_text, assigned_values_text)
combined_survey.to_csv('Phase2Dataset_v2.csv')


# Process text data ###################################################################################################
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import ShuffleSplit
import re
from collections import Counter
from itertools import chain

# download data
survey_comments = pd.read_csv('Phase2Dataset_v2.csv')
survey_comments = survey_comments.iloc[:, 1:]

# df with missing data
cols = ['features', 'other']

survey_comments_unlabled = survey_comments[survey_comments[cols].isna().any(1)]
# df with no missing data
survey_comments = survey_comments[survey_comments[cols].notna().all(1)]
survey_comments.isna().any()
# drop columns with feedback thats just NA
survey_comments = survey_comments.dropna()
survey_comments.isna().any()

# convert date to datetime object
survey_comments['RecordedDate'] = pd.to_datetime(survey_comments['RecordedDate'], format='%m/%d/%Y %H:%M')
survey_comments['Group Satisfaction'] = survey_comments['rating'].apply(
    lambda x: "9-10" if x > 8 else ("1-5" if x < 6 else "6-8"))

# create comment series and convert to proper data type
comments = survey_comments['feedback'].values.astype('U')

# inititialize lemmatization
lemmatizer = WordNetLemmatizer()


# create a function that preprocesses the data
# note may want to do analysis on all caps comments and exclamation points
# add stop words
# create bigrams
def preprocessor(text):
    """ Returns clean text from a string (puts in lower case, removes
        punctuation, removes stop words, removes numbers, and lemmatizes words)

        Parameters
        ----------
        text : str
            text to be cleaned
    """

    text = text.lower()  # text to lower case
    text = re.sub(r'\d+', '', text)  # remove numbers (may remove this)
    text = re.sub(r'\$', ' price ', text)  # may change word

    words = re.split("\\s+", text)  # split text by space before lemmatizer
    stop_words = list(set(stopwords.words('english')) - {'out', 'off'})  # list of english stopwords from nltk
    stop_words = stop_words + ['platform', 'publisher', 'book', 'like']  # add two more words
    words = [w for w in words if w not in stop_words]  # remove stop words

    lemma_words = [lemmatizer.lemmatize(word=word) for word in words]  # lemmatize
    lemma_words = [w for w in lemma_words if len(w) >= 3]  # remove words of certain length

    clean_text = ' '.join(lemma_words)  # add space between words
    clean_text = re.sub(r'[^\w\s]', '', clean_text)  # remove punctuation except $
    clean_text = re.sub(' +', ' ', clean_text)  # remove any additional spaces

    return clean_text


# test the preprocessor
test_string = "this! ma cities$ 5in it is @@@ cc a companies TEST!!@#$%^&*(){}: $1000. heaaasfdfr been had cities"
#print('HEREEEE: ', preprocessor(test_string))

# run preprocessor for all rows (don't do in vectorizer)
comments_processed = pd.Series([preprocessor(row) for row in comments])

# remove infrequent words (can change the number but need this for visualization)
comment_list = comments_processed.str.split().tolist()

# compute global word frequency
c = Counter(chain.from_iterable(comment_list))
c.most_common(50)

# filter, join, and re-assign
survey_comments['clean'] = [' '.join([j for j in i if c[j] > 10]) for i in comment_list]

# remove comments with empty length
survey_comments['CommentLength'] = survey_comments['clean'].str.len()
df_filtered = survey_comments[survey_comments['CommentLength'] > 0]
# reset jndex to sequential numbers
df_filtered.reset_index(drop=True, inplace=True)
clean_comments = df_filtered['clean']

print(df_filtered.head())

#df_filtered.to_csv('filtered_phase2.csv')


# BERT Sentiment Analysis #############################################################################################
# https://towardsdatascience.com/lstm-vs-bert-a-step-by-step-guide-for-tweet-sentiment-analysis-ced697948c47

import nltk
from nltk.tokenize import word_tokenize
def tokenize_text(text):
    """
    Tokenize the text input using NLTKs word_tokenize()
    :param text: the text field the
    :return: tokenized text response
    """
    return [word for word in word_tokenize(text) if (word.isalpha() == 1)]

tokenized_comments = pd.Series([tokenize_text(row) for row in df_filtered])
print(tokenized_comments)

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# summary of the BERT model
#model.summary()

import tensorflow as tf
import pandas as pd

# https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671

# split the data into a training dataset and a test dataset (MAKE SURE IT'S RANDOMIZED
# split data into test and training sets. Shuffly because the data is not random
split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

for train_index, test_index in split.split(df_filtered):
    train_set = df_filtered.iloc[train_index]
    test_set = df_filtered.iloc[test_index]

# trim the dataset for the only necessary columns, the cleaned survery responses and numeric sentiment (-1, 0, 1)
train = train_set[['clean', 'adjusted_rating']].copy()
test = test_set[['clean', 'adjusted_rating']].copy()


# Convert the two pandas dataframes into suitable objects for the BERT model by using the InputExample function
# below calls the InputExample funtion
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

#train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, 'clean', 'adjusted_rating')



def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    """
    tokenizes the survey responses in the InputExample objects
    :param examples:
    :param tokenizer:
    :param max_length:
    :return:
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

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, epochs=2, validation_data=validation_data)











