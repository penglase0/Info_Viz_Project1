# Authors: Courtney Datin and Oliver Penglase
# Course: DS5500 - Information Visualization in Data Science
# Project Phase 1 - Dashboard Code

# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt

# read sentiment csv file in to a dataframe
survey_comments = pd.read_csv('survey_sentiments.csv')

# attempt at dashboard https://docs.streamlit.io/en/stable/getting_started.html
import streamlit as st
st.write("""
# Consumer Review Dashboard
by Courtney Datin and Oliver Penglase
""")

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
