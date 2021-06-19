"""
Created on Mon Jun  7 13:18:05 2021

@author: olliepenglase
"""

import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#import nltk
from sklearn.feature_extraction.text import CountVectorizer
import re
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
import guidedlda
from collections import Counter
from itertools import chain

#nltk.download() #uncomment, run and download all first time running

#%% Data Preprocessing

#download data
survey_comments = pd.read_csv('InfoVizProjectData.csv')



#convert date to datetime object
survey_comments['Date'] =  pd.to_datetime(survey_comments['StartDate'], format='%m/%d/%y %H:%M')

#create comment series and convert to proper data type
comments = survey_comments['OpenResponse'].values.astype('U') 

#inititialize lemmatization
lemmatizer = WordNetLemmatizer()
%%%
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
    lemma_words = [w for w in lemma_words if len(w) >= 3] # remove words of certail length
    
    clean_text = ' '.join(lemma_words) # add space between words
    clean_text=re.sub(r'[^\w\s]','',clean_text) # remove punctuation except $
    clean_text=re.sub(' +', ' ', clean_text) # remove any additional spaces
    
    return clean_text

#test the preprocessor
test_string = "this! ma cities$ 5in is @@@ cc a companies TEST!!@#$%^&*(){}: $1000. heaaasfdfr been had cities"
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

#%% Guided LDA

# stop_words - list of words to remove 
# preprocessor - customizable preprocessor function preprocessor = preprocessor (already run)
vectorizer = CountVectorizer(max_df=0.95, # remove words that appear in 95% of documents
                                min_df=10, # ignore words in only 10 documents
                                ngram_range=(1, 2)) #using a unigram (1,1) or bigram (2,2)

comment_matrix = vectorizer.fit_transform(clean_comments)

## Since LDA is a probabilistic model, you will get some differences in the end 
## result each time you run it. Hence, it is a good idea to set the random_state 
## parameter to a fixed number and save the model locally using pickle to preserve 
## how it infers the topics later on. 


#### Guided LDA
#https://github.com/vi3k6i5/GuidedLDA
#https://medium.com/analytics-vidhya/how-i-tackled-a-real-world-problem-with-guidedlda-55ee803a6f0d

#get vocab of dataset
vocab = vectorizer.get_feature_names()
word_index = dict((word_vocab, index) for index, word_vocab in enumerate(vocab))

# create topic list
topic_keywords = [['log', 'out', 'log out', 'time', 'out', 'time out', 'logged'], #1
['access', 'code', 'access code', 'course', 'course code', 'purchasing', 'purchase', 'purchased', 'subscription', 'bookstore'], #2
['page', 'number', 'page number'], #3
['price', 'expensive', 'pay', 'money', 'save', 'cost', 'overpriced'], #4
['navigate', 'find', 'hard navigate', 'hard find', 'figure out', 'hard use', 'navigation', 'user friendly', 'accessible'], #5
['product', 'rental', 'rent', 'print'], #6
['customer', 'customer service', 'service', 'support', 'customer support'], #7
['app', 'tablet', 'phone', 'mobile', 'ipad'], #8
['load', 'crash', 'glitchy', 'slow', 'bug']] #9

#grid search best priors
alpha = [.001, .01, .1, 1] #distribution over topics
eta = [.001, .01, .1, 1] # distribution over words

seed_topics = {}
for topic_id, topic in enumerate(topic_keywords):
    for word in topic:
        seed_topics[word_index[word]] = topic_id

# own grid due to guidedlda being icompatible with sklearn (must create own scoring)
for i in alpha:
    for j in eta:
        model = guidedlda.GuidedLDA(n_topics=10, n_iter=1000, alpha = i, eta = j, random_state=1, refresh=100)
        model.fit(comment_matrix, seed_topics=seed_topics, seed_confidence=0.15)
        print("\nalpha: ", i, "eta: ", j, "loglikelihood: ", model.loglikelihood(), "\n")

model = guidedlda.GuidedLDA(n_topics=10, n_iter=1000, alpha = .001, eta = .1, random_state=1, refresh=100)
model.fit(comment_matrix, seed_topics=seed_topics, seed_confidence=0.15)

#check top words
n_top_words = 10
topic_word = model.topic_word_ # word topic distribution
for i, topic_dist in enumerate(topic_word): # i is topic topic_dist is distribution of words in topic
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1] #print top 20 words in topic
    print('Topic {}:   {}'.format(i, ' '.join(topic_words)))

# create document topic distribution
doc_topic = model.transform(comment_matrix)

# view words in document
n = comment_matrix.toarray()
doc_topic = model.transform(n)
for i in range(9):
    print("top topic: {} Document: {}".format(doc_topic[i].argmax(),', '.join(np.array(vocab)[list(reversed(n[i,:].argsort()))[0:5]])))

#%% Visualization

#pydavis visualization    
# mds='tsne' 
vis = pyLDAvis.sklearn.prepare(model, comment_matrix, vectorizer, sort_topics=False)
pyLDAvis.enable_notebook()
pyLDAvis.save_html(vis, 'lda.html')


#%% Final Dataframe
#make the dataframe with the final comments

#https://medium.com/@yanlinc/how-to-build-a-lda-topic-model-using-from-text-601cdcbfd3a6
#https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/

# topic names for columns in dataframe
topicnames = ["LoggingIn", "AccessPurchase", "PageNumberSearch", 
              "Price", "Navigation", "Products", "CustomerSupport",
              "OtherDevices", "Technical", "Other"]

# Make the pandas dataframe index=docnames
df_document_topic = pd.DataFrame(np.round(doc_topic, 2), columns=topicnames)
df_document_topic = df_document_topic.applymap(lambda x: 1 if x >= 0.25 else 0)
df_document_topic.sum()/len(df_document_topic)
#df_document_topic["sum"] = df_document_topic.sum(axis=1)
df_merged = df_filtered.merge(df_document_topic, how='outer', left_index=True, right_index=True)
#a = df_merged[df_merged['sum'] == 0]['OpenResponse']


