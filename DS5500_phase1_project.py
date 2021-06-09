#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:18:05 2021

@author: olliepenglase
"""

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn


#nltk.download() #uncomment, run and download all first time running

#download data
survey_comments = pd.read_csv('InfoVizProjectData.csv')

#length of comments possibly for later analysis
survey_comments['CommentLength'] = survey_comments['OpenResponse'].str.len()

#convert date to datetime object
survey_comments['Date'] =  pd.to_datetime(survey_comments['StartDate'], format='%m/%d/%y %H:%M')

#inititialize lemmatization
lemmatizer = WordNetLemmatizer()

#create 
comments = survey_comments['OpenResponse'].values.astype('U') 

a = "this! cities is @@@ a companies TEST!!@#$%^&*(){}: $1000. heaaasfdfr been had cities"

# create a function that preprocesses the data
# note may want to do analysis on all caps comments and exclamation points
# add stop words
def preprocessor(text):
    
    text=text.lower() # text to lower case
    text=re.sub(r'[^\w\s]','',text) #remove punctuation
    #text = re.sub(r'\d+', '', text) #remove numbers (may remove this)
    words=re.split("\\s+",text) #split text by space before lemmatizer
    lemma_words=[lemmatizer.lemmatize(word=word) for word in words]
    return ' '.join(lemma_words)

#test the preprocessor
preprocessor(a)

## You should use CountVectorizer when fitting LDA instead of 
## TfidfVectorizer since LDA is based on term count and document count.


# max_df - remove words that appear in 95% of documents
# min_df - ignore words in only 1 documents
# max_features - ?
# stop_words - list of words to remove 
# ngram_range - using a unigram (1,1) or bigram (2,2)? may want to change to unigram
# preprocessor - customizable preprocessor function
tf_vectorizer = CountVectorizer(max_df=0.90, min_df=1, stop_words='english', ngram_range=(1, 2), preprocessor = preprocessor)
tf = tf_vectorizer.fit_transform(comments)  

#print vocab
#print(tf_vectorizer.get_feature_names())

## Since LDA is a probabilistic model, you will get some differences in the end 
## result each time you run it. Hence, it is a good idea to set the random_state 
## parameter to a fixed number and save the model locally using pickle to preserve 
## how it infers the topics later on. 


# n_components - number of topics
# keep random state to maintain the same topics (LDA is probabilistic model)
lda = LatentDirichletAllocation(random_state=1, n_components = 20).fit(tf)
    
# print top words in each topic
for index, topic in enumerate(lda.components_):
    print(f'Top 20 words for Topic #{index}')
    print([tf_vectorizer.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print('\n')
    
    
vis = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
pyLDAvis.enable_notebook()
viz = pyLDAvis.display(vis)


pyLDAvis.save_html(vis, 'lda.html')




#This part needs work

#https://medium.com/@yanlinc/how-to-build-a-lda-topic-model-using-from-text-601cdcbfd3a6
#https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
# index names
docnames = ["Doc" + str(i) for i in range(len(survey_comments))]

topicnames = ["Topic" + str(i) for i in range(lda.n_components)]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda, 2), columns=topicnames, index=docnames)

