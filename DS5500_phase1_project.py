
## ---(Wed Jun 16 14:39:43 2021)---


"""
Created on Mon Jun  7 13:18:05 2021

@author: olliepenglase
"""

import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import re
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
import guidedlda


#nltk.download() #uncomment, run and download all first time running

#download data
survey_comments = pd.read_csv('InfoVizProjectData.csv')

#length of comments possibly for later analysis
survey_comments['CommentLength'] = survey_comments['OpenResponse'].str.len()

#convert date to datetime object
survey_comments['Date'] =  pd.to_datetime(survey_comments['StartDate'], format='%m/%d/%y %H:%M')

#create comment series and convert to proper data type
comments = survey_comments['OpenResponse'].values.astype('U') 

#inititialize lemmatization
lemmatizer = WordNetLemmatizer()

a = "this! ma cities is @@@ cc a companies TEST!!@#$%^&*(){}: $1000. heaaasfdfr been had cities"

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
    text = re.sub('\$', ' price ', text) # may change word
    words=re.split("\\s+",text) # split text by space before lemmatizer
    stop_words = list(set(stopwords.words('english')) - {'out', 'off'}) # list of english stopwords from nltk
    stop_words = stop_words + ['platform', 'publisher', 'book'] # add two more words
    words = [w for w in words if w not in stop_words] # remove stop words
    lemma_words=[lemmatizer.lemmatize(word=word) for word in words] #lemmatize
    lemma_words = [w for w in lemma_words if len(w) >= 3] # remove words of certail length
    clean_text = ' '.join(lemma_words) # add space between words
    clean_text=re.sub(r'[^\w\s$]','',clean_text) #remove punctuation except $
    clean_text=re.sub(' +', ' ', clean_text)
    
    return clean_text

#test the preprocessor
a = "this! ma cities 5in is @@@ cc a companies TEST!!@#$%^&*(){}: $1000. heaaasfdfr been had cities"
preprocessor(a)

a = [preprocessor(row) for row in comments]

survey_comments['clean'] = a 
survey_comments['CommentLength'] = survey_comments['clean'].str.len()
df_filtered = survey_comments[survey_comments['CommentLength'] > 0]
a = df_filtered['clean']   
## You should use CountVectorizer when fitting LDA instead of 
## TfidfVectorizer since LDA is based on term count and document count.


# max_df - remove words that appear in 95% of documents
# min_df - ignore words in only 1 documents
# max_features - ?
# stop_words - list of words to remove 
# ngram_range - using a unigram (1,1) or bigram (2,2)? may want to change to unigram
# preprocessor - customizable preprocessor function
vectorizer = CountVectorizer(max_df=0.95, 
                                min_df=10, 
                                ngram_range=(1, 2), 
                                preprocessor = preprocessor)

tf = vectorizer.fit_transform(a)
#print vocab
#print(vectorizer.get_feature_names())

#vectorizer.vocabulary_

## Since LDA is a probabilistic model, you will get some differences in the end 
## result each time you run it. Hence, it is a good idea to set the random_state 
## parameter to a fixed number and save the model locally using pickle to preserve 
## how it infers the topics later on. 


# n_components - number of topics
# keep random state to maintain the same topics (LDA is probabilistic model)
lda = LatentDirichletAllocation(random_state=1)
#lda.score(tf)


search_params = {'n_components': [10]}
model = GridSearchCV(lda, param_grid=search_params)
model.fit(tf)


#### Guided LDA
tf_feature_names = vectorizer.get_feature_names()
word2id = dict((v, idx) for idx, v in enumerate(tf_feature_names))

seed_topic_list= [['log', 'out', 'log out', 'time', 'out', 'time out', 'logged'], #1
['access', 'code', 'access code', 'course', 'course code', 'purchasing', 'purchase', 'purchased', 'subscription', 'bookstore'], #2
['page', 'number', 'page number'], #3
['price', 'expensive', 'pay', 'money', 'save', 'cost', 'overpriced'], #4
['navigate', 'find', 'hard navigate', 'hard find', 'figure out', 'hard use', 'navigation', 'user friendly'], #5
['product', 'rental', 'rent', 'print'], #6
['customer', 'customer service', 'service', 'support', 'customer support'], #7
['app', 'tablet', 'phone', 'mobile'], #8
['load', 'crash', 'glitchy', 'slow', 'bug']] #9

model = guidedlda.GuidedLDA(n_topics=10, n_iter=1000, random_state=1, refresh=10)
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id
model.fit(tf, seed_topics=seed_topics, seed_confidence=0.3)

#check top words
n_top_words = 15
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
     topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
     print('Topic {}: {}'.format(i, ' '.join(topic_words)))

# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(tf))




# See model parameters
print(lda.get_params())

# print top words in each topic
for index, topic in enumerate(best_lda_model.components_):
    print(f'Top 20 words for Topic #{index}')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print('\n')
    

#pydavis visualization    
# mds='tsne' 
vis = pyLDAvis.sklearn.prepare(model, tf, vectorizer)
pyLDAvis.enable_notebook()
pyLDAvis.save_html(vis, 'lda.html')




#This part needs work

#https://medium.com/@yanlinc/how-to-build-a-lda-topic-model-using-from-text-601cdcbfd3a6
#https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
# index names
docnames = [i for i in range(len(df_filtered))]

topicnames = ["Topic" + str(i) for i in range(10)]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(model.doc_topic_, 2), columns=topicnames, index=docnames)
df_document_topic["sum"] = df_document_topic.sum(axis=1)
df_merged = survey_comments.merge(df_document_topic, how='outer', left_index=True, right_index=True)


