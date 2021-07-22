"""
Created on Mon Jun  7 13:18:05 2021

@author: olliepenglase, Courtney Datin
"""

import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
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

#%% non BERT algorithms

#split data into test and training sets
split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

for train_index, test_index in split.split(df_filtered):
    train_set = df_filtered.iloc[train_index]
    test_set = df_filtered.iloc[test_index]

y_train = train_set.iloc[:, 5:12]
x_train = train_set['clean']
vetorizar = TfidfVectorizer(max_features=3000, max_df=0.85)
vetorizar.fit(x_train)
x_train_tfidf = vetorizar.transform(x_train)

svm = LinearSVC(random_state=42)
multilabel_classifier = MultiOutputClassifier(svm, n_jobs=-1)
multilabel_classifier = multilabel_classifier.fit(x_train_tfidf, y_train)
y_train_pred = multilabel_classifier.predict(x_train_tfidf)
matrices = multilabel_confusion_matrix(y_train, y_train_pred)

cmd = ConfusionMatrixDisplay(matrices[0], display_labels=np.unique(y_train)).plot()
plt.title('Confusion Matrix for label 1 (type)')
plt.show()
cmd = ConfusionMatrixDisplay(matrices[1], display_labels=np.unique(y_train)).plot()
plt.title('Confusion Matrix for label 2 (color)')
plt.show()

pipe_svm = Pipeline([
   ('tfidf', TfidfVectorizer(analyzer='word', 
                                  ngram_range = (1,2), 
                                  max_df = 0.95, 
                                  min_df = 10,
                                  preprocessor = preprocessor)),
    ('clf_svm', LinearSVC(random_state=0, C=10)),
])
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






