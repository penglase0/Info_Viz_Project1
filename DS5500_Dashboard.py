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
print(survey_comments.head())

# convert StartDate column to be recognized as a date
survey_comments['StartDate'] = pd.to_datetime(survey_comments['StartDate'])

# create a more generalized date field for digestability of the dashboard
survey_comments['month'] = survey_comments['StartDate'].dt.strftime('%m %Y')
print(survey_comments.head())
# Streamlit documentation: https://docs.streamlit.io/en/stable/getting_started.html

# dashboard header and authors
import streamlit as st
st.write("""
# Consumer Review Dashboard
by Courtney Datin and Oliver Penglase
""")

st.write("""
## Lifetime Analysis""")

## create line graph of % positive reviews over time
# aggregate to get total reviews by month
survey_comments['constant'] = 1
responses_by_month = survey_comments.groupby(['month']).sum()['constant'].to_frame().reset_index()
print(responses_by_month.head())

# aggregate to get % positive reviews by month
positive_condition = survey_comments['compound_sentiment'] == 'positive'
positive_responses = survey_comments[positive_condition].groupby(['month']).sum()['constant'].to_frame().reset_index()
print(positive_responses.head())

# merge dataframes together and calculate percentage
responses_by_month = pd.merge(responses_by_month, positive_responses, on='month')
responses_by_month.columns = ['month', 'total_responses', 'positive_responses']
responses_by_month['percent_positive'] = responses_by_month['positive_responses']/responses_by_month['total_responses']
print(responses_by_month.head())

# graph bar chart
total_time_bar = st.line_chart(responses_by_month[['month', 'percent_positive']].set_index('month'))


st.write(""" 
## Monthly Analysis""")

# drop down menu to select timeframe for filtering data
month_list = ['08 2018',
              '09 2018',
              '10 2018',
              '11 2018',
              '12 2018',
              '01 2019',
              '02 2019',
              '03 2019',
              '04 2019',
              '05 2019',
              '06 2019',
              '07 2019',
              '08 2019',
              '09 2019',
              '10 2019',
              '11 2019',
              '12 2019',
              '01 2020',
              '02 2020',
              '03 2020',
              '04 2020',
              '05 2020',
              '06 2020',
              '07 2020',
              '08 2020',
              '09 2020',
              '10 2020',
              '11 2020',
              '12 2020']

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



