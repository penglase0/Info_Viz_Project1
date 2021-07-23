"""
Created on Mon Jun  7 13:18:05 2021

@author: olliepenglase, Courtney Datin
"""

import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn import metrics
import re
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
#nltk.download() #uncomment, run and download all first time running
#nltk.download('vader_lexicon')
#%% Data Preprocessing

#download data
survey_comments = pd.read_csv('~/Phase2Data/Phase2Dataset_v2.csv')
survey_comments = survey_comments.iloc[: , 1:]

# df with missing data
cols = ['features', 'other']

survey_comments_unlabled = survey_comments[survey_comments[cols].isna().any(1)]
# df with no missing data
survey_comments = survey_comments[survey_comments[cols].notna().all(1)]
survey_comments.isna().any()
# drop columns with feedback thats just NA
survey_comments = survey_comments.dropna()
survey_comments.isna().any()


#convert date to datetime object
survey_comments['RecordedDate'] =  pd.to_datetime(survey_comments['RecordedDate'], format='%m/%d/%Y %H:%M')
survey_comments['Group Satisfaction'] = survey_comments['rating'].apply(lambda x: "9-10" if x > 8 else("1-5" if x < 6 else "6-8"))


#create comment series and convert to proper data type
comments = survey_comments['feedback'].values.astype('U') 

#inititialize lemmatization
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
    
    text=text.lower() # text to lower case
    text = re.sub(r'\d+', '', text) # remove numbers (may remove this)
    text = re.sub(r'\$', ' price ', text) # may change word
   
    words=re.split("\\s+",text) # split text by space before lemmatizer
    stop_words = list(set(stopwords.words('english')) - {'out', 'off'}) # list of english stopwords from nltk
    stop_words = stop_words + ['platform', 'publisher', 'book', 'like'] # add two more words
    words = [w for w in words if w not in stop_words] # remove stop words
    
    lemma_words=[lemmatizer.lemmatize(word=word) for word in words] #lemmatize
    lemma_words = [w for w in lemma_words if len(w) >= 3] # remove words of certain length
    
    clean_text = ' '.join(lemma_words) # add space between words
    clean_text=re.sub(r'[^\w\s]','',clean_text) # remove punctuation except $
    clean_text=re.sub(' +', ' ', clean_text) # remove any additional spaces
    
    return clean_text

#test the preprocessor
test_string = "this! ma cities$ 5in it is @@@ cc a companies TEST!!@#$%^&*(){}: $1000. heaaasfdfr been had cities"
preprocessor(test_string)

#run preprocessor for all rows (don't do in vectorizer)
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

#%% non BERT algorithms

#https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5

#split data into test and training sets. Shuffly because the data is not random
split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

for train_index, test_index in split.split(df_filtered):
    train_set = df_filtered.iloc[train_index]
    test_set = df_filtered.iloc[test_index]

y_train = train_set.iloc[:, 5:13]
x_train = train_set['clean']
y_test = train_set.iloc[:, 5:13]
x_test= train_set['clean']


pipe_svm = Pipeline([
   ('tfidf', TfidfVectorizer(analyzer='word', 
                                  ngram_range = (1,2), 
                                  max_df = 0.95, 
                                  min_df = 10,
                                  preprocessor = preprocessor)),
    ('clf_svm', OneVsRestClassifier(LinearSVC(random_state=0, C=10))),
])


categories = list(train_set.iloc[:, 5:13].columns)


for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    pipe_svm.fit(x_train, y_train[category])
    # compute the testing accuracy
    prediction = pipe_svm.predict(x_test)
    cat_name_pred = category + "_pred"
    train_set[cat_name_pred] = prediction
    print('Test accuracy is {}'.format(accuracy_score(y_test[category], prediction)))


