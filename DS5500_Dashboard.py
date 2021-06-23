# Authors: Courtney Datin and Oliver Penglase
# Course: DS5500 - Information Visualization in Data Science
# Project Phase 1 - Dashboard Code

# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

import streamlit
import streamlit as st

# read sentiment csv file in to a dataframe
survey_comments = pd.read_csv('final_df.csv')

# convert StartDate column to be recognized as a date
survey_comments['StartDate'] = pd.to_datetime(survey_comments['StartDate'])

# create a more generalized date field for digestability of the dashboard
survey_comments['month'] = survey_comments['StartDate'].dt.strftime('%B %Y')

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

# aggregate to get % positive reviews by month
positive_condition = survey_comments['compound_sentiment'] == 'positive'
positive_responses = survey_comments[positive_condition].groupby(['month']).sum()['constant'].to_frame().reset_index()

# merge dataframes together and calculate percentage
responses_by_month = pd.merge(responses_by_month, positive_responses, on='month')
responses_by_month.columns = ['month', 'total_responses', 'positive_responses']
responses_by_month['percent_positive'] = responses_by_month['positive_responses']/responses_by_month['total_responses']
print(responses_by_month.head())

# format month column so it's aligned properly in the chart
responses_by_month['month'] = pd.to_datetime(responses_by_month['month'])
responses_by_month = responses_by_month.sort_values(by='month')

# graph line graph of the percentage of positive reviews by month
st.write("Pertentage of Positive Reviews by Month")
total_time_bar = st.line_chart(responses_by_month[['month', 'percent_positive']].set_index('month'))



## add horizontal bar chart by topic for user specified month
# plot pie chart of polarities
survey_comments['constant'] = 1
sentiment_plot = survey_comments.groupby(['compound_sentiment']).sum()['constant'].to_frame()
#sentiment_plot.plot.pie(y='constant', autopct='%1.1f%%', startangle=90)
#plt.title('Student Sentiment', fontsize=22)

sentiment_plot = survey_comments.groupby(['compound_sentiment'])["LoggingIn", "AccessPurchase", "PageNumberSearch",
                                                           "Price", "Navigation", "Products", "CustomerSupport",
                                                           "OtherDevices", "Technical", "Other"].sum()
sentiment_plot = sentiment_plot / sentiment_plot.sum()
sentiment_plot = sentiment_plot.T.sort_values(by=['positive'])
print(sentiment_plot.head())
for col in sentiment_plot.columns:
    print(col)

negative = sentiment_plot['negative'].tolist()
neutral = sentiment_plot['neutral'].tolist()
positive = sentiment_plot['positive'].tolist()
y = sentiment_plot.index.tolist()

fig_all_time = plt.figure()
plt.barh(y, negative, color = 'red')
plt.barh(y, neutral, color = 'yellow', left = negative)
plt.barh(y, positive, color = 'green', left=list(map(lambda neg, neu: neg + neu, negative, neutral)))

st.write("All Time Sentiment Percentage by Topic")
st.pyplot(fig_all_time)

################################################################################################

st.write(""" 
## Monthly Analysis""")

# drop down menu to select timeframe for filtering data
month_list = ['August 2018',
              'September 2018',
              'October 2018',
              'November 2018',
              'December 2018',
              'January 2019',
              'February 2019',
              'March 2019',
              'April 2019',
              'May 2019',
              'June 2019',
              'July 2019',
              'August 2019',
              'September 2019',
              'October 2019',
              'November 2019',
              'December 2019',
              'January 2020',
              'February 2020',
              'March 2020',
              'April 2020',
              'May 2020',
              'June 2020',
              'July 2020',
              'August 2020',
              'September 2020',
              'October 2020',
              'November 2020',
              'December 2020']

month_year_option = st.selectbox('Select a month to filter the Dashboard:', month_list)
#'Month selected: ', month_year_option
print(month_year_option)
print(type(month_year_option))

date_requirement = survey_comments['month'] == month_year_option
date_filtered_survey = survey_comments[date_requirement]



## bar chart showing all-time number of negative, neutral, and positive reviews
# aggregate data by sentiment and create a bar chart
date_filtered_survey['constant'] = 1
sentiment_plot = date_filtered_survey.groupby(['compound_sentiment']).sum()['constant'].to_frame()
st.write('Month Survey Volume by Sentiment')
st.bar_chart(sentiment_plot)



## add horizontal bar chart by topic for user specified month
# plot pie chart of polarities
date_filtered_survey['constant'] = 1
sentiment_plot = date_filtered_survey.groupby(['compound_sentiment']).sum()['constant'].to_frame()
#sentiment_plot.plot.pie(y='constant', autopct='%1.1f%%', startangle=90)
#plt.title('Student Sentiment', fontsize=22)

sentiment_plot = date_filtered_survey.groupby(['compound_sentiment'])["LoggingIn", "AccessPurchase", "PageNumberSearch",
                                                           "Price", "Navigation", "Products", "CustomerSupport",
                                                           "OtherDevices", "Technical", "Other"].sum()
sentiment_plot = sentiment_plot / sentiment_plot.sum()
sentiment_plot = sentiment_plot.T.sort_values(by=['positive'])
print(sentiment_plot.head())
for col in sentiment_plot.columns:
    print(col)

negative = sentiment_plot['negative'].tolist()
neutral = sentiment_plot['neutral'].tolist()
positive = sentiment_plot['positive'].tolist()
y = sentiment_plot.index.tolist()

fig_monthly_bar = plt.figure()
plt.barh(y, negative, color = 'red')
plt.barh(y, neutral, color = 'yellow', left = negative)
plt.barh(y, positive, color = 'green', left=list(map(lambda neg, neu: neg + neu, negative, neutral)))

st.write("Month Sentiment Percentage by Topic")
st.pyplot(fig_monthly_bar)



# remove unnecessary columns and display dataframe
survey_condensed = date_filtered_survey[['OverallSatisfaction', 'OpenResponse', 'compound', 'compound_sentiment']].copy()
st.dataframe(data=survey_condensed)



