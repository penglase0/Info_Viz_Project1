# Authors: Courtney Datin and Oliver Penglase
# Course: DS5500 - Information Visualization in Data Science
# Project Phase 1

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

# add sentiment scores to dataframe
survey_comments['sentiment_scores'] = survey_comments['OpenResponse'].apply(lambda OpenResponse: sid.polarity_scores(OpenResponse))
#print(survey_comments.head())

# extract compound column (normalized sum of lexicon ratings)
survey_comments['compound'] = survey_comments['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])
#print(survey_comments.head())

# categorize the compound as negative or positive
survey_comments['compound_sentiment'] = survey_comments['compound'].apply(lambda x: 'positive' if x >= 0 else 'negative')
#print(survey_comments.head())

#print(survey_comments.to_markdown())

# plot pie chart of polarities
survey_comments['constant'] = 1
sentiment_plot = survey_comments.groupby(['compound_sentiment']).sum()['constant'].to_frame()
sentiment_plot.plot.pie(y='constant', autopct='%1.1f%%', startangle=90)
plt.title('Proportion of Sentiments', fontsize=22)
plt.show()





