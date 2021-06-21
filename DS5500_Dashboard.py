# Authors: Courtney Datin and Oliver Penglase
# Course: DS5500 - Information Visualization in Data Science
# Project Phase 1 - Dashboard Code

# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import streamlit as st

# read sentiment csv file in to a dataframe
survey_comments = pd.read_csv('final_df.csv')

# convert StartDate column to be recognized as a date
survey_comments['StartDate'] = pd.to_datetime(survey_comments['StartDate'])

# create a more generalized date field for digestability of the dashboard
survey_comments['month'] = survey_comments['StartDate'].dt.strftime('%b %Y')

# Streamlit documentation: https://docs.streamlit.io/en/stable/getting_started.html

# dashboard header and authors
import streamlit as st
st.write("""
# Consumer Review Dashboard
by Courtney Datin and Oliver Penglase
""")

st.write("""
## Lifetime Analysis""")


st.write(""" 
## Monthly Analysis""")

# drop down menu to select timeframe for filtering data
month_list = ['Aug 2018',
              'Sep 2018',
              'Oct 2018',
              'Nov 2018',
              'Dec 2018',
              'Jan 2019',
              'Feb 2019',
              'Mar 2019',
              'Apr 2019',
              'May 2019',
              'Jun 2019',
              'Jul 2019',
              'Aug 2019',
              'Sep 2019',
              'Oct 2019',
              'Nov 2019',
              'Dec 2019',
              'Jan 2020',
              'Feb 2020',
              'Mar 2020',
              'Apr 2020',
              'May 2020',
              'Jun 2020',
              'Jul 2020',
              'Aug 2020',
              'Sep 2020',
              'Oct 2020',
              'Nov 2020',
              'Dec 2020']

month_year_option = st.selectbox('Select a month to filter the Dashboard:', month_list)
'Month selected: ', month_year_option
print(month_year_option)
print(type(month_year_option))

test = 'Aug 2018'

date_requirement = survey_comments['month'] == month_year_option

#date_filtered_survey = survey_comments[survey_comments['month'] = month_year_option]
date_filtered_survey = survey_comments[date_requirement]

# bar chart showing all-time number of negative, neutral, and positive reviews
# aggregate data by sentiment and create a bar chart
date_filtered_survey['constant'] = 1
sentiment_plot = date_filtered_survey.groupby(['compound_sentiment']).sum()['constant'].to_frame()
st.bar_chart(sentiment_plot)

# remove unnessecary columns and display dataframe
survey_condensed = date_filtered_survey[['OverallSatisfaction', 'OpenResponse', 'compound', 'compound_sentiment']].copy()
st.dataframe(data=survey_condensed)

# aggregate to get % of negative reviews by month for a line graph

