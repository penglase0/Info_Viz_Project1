# Authors: Courtney Datin and Oliver Penglase
# Course: DS5500 - Information Visualization in Data Science
# Project Phase 1 - Sentiment Analysis Code

# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import tabulate
import nltk
nltk.download('vader_lexicon')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# read in csv dataset as a dataframe
survey_comments = pd.read_csv('InfoVizProjectData.csv')

# process the data (remove NAs and blank spaces)
survey_comments.dropna(inplace=True)

test = 'this product is amazing and has been very useful'
print(sid.polarity_scores(test))


# add sentiment scores to dataframe
survey_comments['sentiment_scores'] = survey_comments['OpenResponse'].apply(
    lambda OpenResponse: sid.polarity_scores(OpenResponse))

# extract compound column (normalized sum of lexicon ratings)
survey_comments['compound'] = survey_comments['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

# categorize the compound as negative or positive
compound_sentiment = [] # define an empty list

for i in survey_comments['compound']: # iterate through compound values and assign sentiment
    if i >= 0.2:
        temp_sentiment = 'positive'
    elif i <= -0.2:
        temp_sentiment = 'negative'
    else:
        temp_sentiment = 'neutral'

    compound_sentiment.append(temp_sentiment)

survey_comments['compound_sentiment'] = compound_sentiment # add the defined sentiments to dataframe

# save survey comments as dataframe for dashboard code
survey_comments.to_csv('survey_sentiments')

#print(survey_comments.head())

#print(survey_comments.to_markdown())

# plot pie chart of polarities
survey_comments['constant'] = 1
sentiment_plot = survey_comments.groupby(['compound_sentiment']).sum()['constant'].to_frame()
sentiment_plot.plot.pie(y='constant', autopct='%1.1f%%', startangle=90)
plt.title('All Time Review Sentiments', fontsize=22)
#plt.show()

print(sentiment_plot.head())


#sentiment_plot.plot.bar(x='compound_sentiment', height='constant')
#print(type(survey_comments)) # it's a dataframe
#comments_sorted = survey_comments.sort_values(['sentiment_scores', 'OpenResponse'], ascending=True)
#print(comments_sorted.head())

# attempt at dashboard https://docs.streamlit.io/en/stable/getting_started.html
import streamlit as st
st.write("""
# Consumer Review Dashboard
by Courtney Datin and Oliver Penglase
""")

st.bar_chart(sentiment_plot)
survey_condensed = survey_comments[['OverallSatisfaction', 'OpenResponse', 'compound', 'compound_sentiment']].copy()
st.dataframe(data=survey_condensed)










