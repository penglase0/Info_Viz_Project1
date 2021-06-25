"""
Created on Mon Jun  7 13:18:05 2021

@author: olliepenglase, Courtney Datin
"""

import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
import guidedlda
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
#nltk.download() #uncomment, run and download all first time running
#nltk.download('vader_lexicon')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#%% Data Preprocessing

#download data
survey_comments = pd.read_csv('InfoVizProjectData.csv')

#convert date to datetime object
survey_comments['StartDate'] =  pd.to_datetime(survey_comments['StartDate'], format='%m/%d/%y %H:%M')

#create comment series and convert to proper data type
comments = survey_comments['OpenResponse'].values.astype('U') 

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
    lemma_words = [w for w in lemma_words if len(w) >= 3] # remove words of certail length
    
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

#%% Guided LDA

# stop_words - list of words to remove 
# preprocessor - customizable preprocessor function preprocessor = preprocessor (already run)
vectorizer = CountVectorizer(max_df=0.95, # remove words that appear in 95% of documents
                                min_df=5, # ignore words in only 10 documents
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
['load', 'crash', 'glitchy', 'slow', 'bug', 'flash', 'glitch']] #9

#grid search best priors
alpha = [.001, .01, .1] #distribution over topics
eta = [.001, .01, .1] # distribution over words

seed_topics = {}
for topic_id, topic in enumerate(topic_keywords):
    for word in topic:
        seed_topics[word_index[word]] = topic_id

# own grid due to guidedlda being icompatible with sklearn (must create own scoring)
#for i in alpha:
#    for j in eta:
#        model = guidedlda.GuidedLDA(n_topics=10, n_iter=1000, alpha = i, eta = j, random_state=1, refresh=100)
#        model.fit(comment_matrix, seed_topics=seed_topics, seed_confidence=0.15)
#        print("\nalpha: ", i, "eta: ", j, "loglikelihood: ", model.loglikelihood(), "\n")

model = guidedlda.GuidedLDA(n_topics=10, n_iter=1000, alpha = .001, eta = .1, random_state=1, refresh=100)
model.fit(comment_matrix, seed_topics=seed_topics, seed_confidence=0.3)

#check top words
n_top_words = 10
topic_word = model.topic_word_ # word topic distribution
for i, topic_dist in enumerate(topic_word): # i is topic topic_dist is distribution of words in topic
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1] #print top 20 words in topic
    print('Topic {}:   {}'.format(i, ', '.join(topic_words)))

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
pyLDAvis.save_html(vis, 'lda_new.html')

#%% Final Dataframe
#make the dataframe with the final comments

#https://medium.com/@yanlinc/how-to-build-a-lda-topic-model-using-from-text-601cdcbfd3a6
#https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/

# topic names for columns in dataframe
topicnames = ["LoggingIn", "AccessPurchase", "PageNumberSearch", 
              "Price", "Navigation", "Products", "CustomerSupport",
              "OtherDevices", "Technical", "Other(Homework)"]

# Make the pandas dataframe index=docnames
df_document_topic = pd.DataFrame(np.round(doc_topic, 2), columns=topicnames)
df_document_topic = df_document_topic.applymap(lambda x: 1 if x >= 0.25 else 0)
df_document_topic.sum()/len(df_document_topic)
#df_document_topic["sum"] = df_document_topic.sum(axis=1)
df_merged = df_filtered.merge(df_document_topic, how='outer', left_index=True, right_index=True)
#a = df_merged[df_merged['sum'] == 0]['OpenResponse']
df_merged['Group Satisfaction'] = df_merged['OverallSatisfaction'].apply(lambda x: "9-10" if x > 8 else("1-5" if x < 6 else "6-8"))


#%% Sentiment Analysis

sid = SentimentIntensityAnalyzer()

# add sentiment scores to dataframe
df_merged['sentiment_scores'] = df_merged['OpenResponse'].apply(
    lambda OpenResponse: sid.polarity_scores(OpenResponse))

# extract compound column (normalized sum of lexicon ratings)
df_merged['compound'] = df_merged['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

# categorize the compound as negative or positive
compound_sentiment = [] # define an empty list

for i in df_merged['compound']: # iterate through compound values and assign sentiment
    if i >= 0.2:
        temp_sentiment = 'positive'
    elif i <= -0.2:
        temp_sentiment = 'negative'
    else:
        temp_sentiment = 'neutral'

    compound_sentiment.append(temp_sentiment)

df_merged['sentiment'] = compound_sentiment # add the defined sentiments to dataframe

survey_condensed = df_merged[['ResponseId', 'StartDate', 'OverallSatisfaction', 
                                         'OpenResponse','LoggingIn','AccessPurchase', 
                                         'PageNumberSearch', 'Price', 'Navigation', 'Products',
                                         'CustomerSupport', 'OtherDevices', 'Technical', 'Other(Homework)',
                                         'Group Satisfaction', 'sentiment']].copy()
# save survey comments as dataframe for dashboard code
survey_condensed.to_csv('final_df.csv')

#%% Simple Analysis

# Topic dist by date

topic_date = df_merged.groupby(pd.DatetimeIndex(df_merged['StartDate']).year)[topicnames].sum()
topic_date["sum"] = topic_date.sum(axis=1)
df_new = topic_date.loc[:,topicnames].div(topic_date["sum"], axis=0)

plt.style.use('ggplot')
ax = df_new.T.plot.barh(rot=0, figsize=(9,6))
ax.set_xlabel('Topic Distribution by Year')
ax.set_title('Percent of Comments in Topic For a Given Year')

# Satisfaction by topic

#df_merged.groupby(['name', 'id', 'dept'])['total_sale'].mean().reset_index()
topic_sat = [sum(df_merged[i]*df_merged["OverallSatisfaction"])/sum(df_merged[i]) for i in topicnames]
topic_sat_sorted = zip(topic_sat, topicnames)
sorted_pairs = sorted(topic_sat_sorted)
tuples = zip(*sorted_pairs)
topic_sat, topicnames = [list(tuple) for tuple in tuples]

y_pos = np.arange(len(topicnames))

plt.style.use('ggplot')
fig, ax = plt.subplots()

hbars = ax.barh(y_pos, topic_sat)
ax.set_yticks(y_pos)
ax.set_yticklabels(topicnames)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Average Satisfaction')
ax.set_title('Avg. Satisfaction by Topic')
# Label with specially formatted floats
for i, v in enumerate(topic_sat):
    ax.text(v + .05, i + .2, round(v,2), color='blue', fontweight='bold')

# Sentiment plots

# plot pie chart of polarities
df_merged['constant'] = 1
sentiment_plot = df_merged.groupby(['sentiment']).sum()['constant'].to_frame()
sentiment_plot.plot.pie(y='constant', autopct='%1.1f%%', startangle=90)
plt.title('Student Sentiment', fontsize=22)

sentiment_plot = df_merged.groupby(['sentiment'])["LoggingIn", "AccessPurchase", "PageNumberSearch", 
              "Price", "Navigation", "Products", "CustomerSupport",
              "OtherDevices", "Technical", "Other(Homework)"].sum()
sentiment_plot = sentiment_plot/sentiment_plot.sum()
sentiment_plot = sentiment_plot.T.sort_values(by=['positive'])

ax = sentiment_plot.plot.barh(stacked=True, color={"negative": "red", "neutral": "yellow", "positive": "green"})
ax.set_xlabel('Percent of Comments with Negative, Neutral, or Positive Sentiment')
ax.set_title('Sentiment by Topic')
ax.legend(loc='upper right', framealpha=1.0)

