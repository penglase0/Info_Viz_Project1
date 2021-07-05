#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:57:30 2021

@author: olliepenglase
"""
# Authors: Courtney Datin and Oliver Penglase
# Course: DS5500 - Information Visualization in Data Science
# Project Phase 1 - Dashboard Code

# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import streamlit
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

# topic names in Dataframe
topicnames = ["LoggingIn", "AccessPurchase", "PageNumberSearch", 
              "Price", "Navigation", "Products", "CustomerSupport",
              "OtherDevices", "Technical", "Other(Homework)"]

# read sentiment csv file in to a dataframe
survey_comments = pd.read_csv('final_df_v2.csv')
del survey_comments['Unnamed: 0']

# convert StartDate column to be recognized as a date
survey_comments['StartDate'] = pd.to_datetime(survey_comments['StartDate'])

# create a more generalized date field for digestability of the dashboard
survey_comments['month'] = survey_comments['StartDate'].dt.strftime('%B %Y')

# Streamlit documentation: https://docs.streamlit.io/en/stable/getting_started.html

# dashboard header and authors

st.write("""
# Consumer Review Dashboard
by Courtney Datin and Oliver Penglase
""")

# add break line and create lifetime analysis header
st.markdown('---')
lifetime_analysis = '<p style="color:Gray; font-size: 30px;"> Lifetime Analysis</p>'
st.markdown(lifetime_analysis, unsafe_allow_html=True)

## create line graph of % positive reviews over time
# aggregate to get total reviews by month
survey_comments['constant'] = 1
responses_by_month = survey_comments.groupby(['month']).sum()['constant'].to_frame().reset_index()

# aggregate to get % positive reviews by month
positive_condition = survey_comments['sentiment'] == 'positive'
positive_responses = survey_comments[positive_condition].groupby(['month']).sum()['constant'].to_frame().reset_index()

# merge dataframes together and calculate percentage
responses_by_month = pd.merge(responses_by_month, positive_responses, on='month')
responses_by_month.columns = ['month', 'total_responses', 'positive_responses']
responses_by_month['% Positive Sentiment'] = responses_by_month['positive_responses']/responses_by_month['total_responses']

# format month column so it's aligned properly in the chart
responses_by_month['month'] = pd.to_datetime(responses_by_month['month'])
responses_by_month = responses_by_month.sort_values(by='month')

# graph line graph of the percentage of positive reviews by month
#st.write("Pertentage of Positive Reviews by Month")
#total_time_bar = st.line_chart(responses_by_month[['month', '% Positive Sentiment']].set_index('month'))

positive_perc_line_graph = '*Percentage of Positive Reviews by Month*'
st.markdown(positive_perc_line_graph)
total_time_bar = st.line_chart(responses_by_month[['month', '% Positive Sentiment']].set_index('month'))
st.text("")
st.text("")

## add horizontal bar chart by topic for user specified month
# plot pie chart of polarities
survey_comments['constant'] = 1
sentiment_plot = survey_comments.groupby(['sentiment']).sum()['constant'].to_frame()
#sentiment_plot.plot.pie(y='constant', autopct='%1.1f%%', startangle=90)
#plt.title('Student Sentiment', fontsize=22)

sentiment_plot = survey_comments.groupby(['sentiment'])[topicnames].sum()
sentiment_plot = sentiment_plot / sentiment_plot.sum()
sentiment_plot = sentiment_plot.T.sort_values(by=['positive'])
for col in sentiment_plot.columns:
    print(col)

negative = sentiment_plot['negative'].tolist()
neutral = sentiment_plot['neutral'].tolist()
positive = sentiment_plot['positive'].tolist()
y = sentiment_plot.index.tolist()

fig1 = plt.figure(figsize=(9,6))
plt.barh(y, negative, color = 'red')
plt.barh(y, neutral, color = 'yellow', left = negative)
plt.barh(y, positive, color = 'green', left=list(map(lambda neg, neu: neg + neu, negative, neutral)))
plt.title("All Time Sentiment Percentage by Topic")
st.pyplot(fig1)

# Topic dist by date

topic_date = survey_comments.groupby(pd.DatetimeIndex(survey_comments['StartDate']).year)[topicnames].sum()
topic_date["sum"] = topic_date.sum(axis=1)
df_new = topic_date.loc[:,topicnames].div(topic_date["sum"], axis=0)

plt.style.use('ggplot')
fig2, ax = plt.subplots(figsize=(9,6))
df_new.T.plot.barh(ax =ax, rot=0, figsize=(9,6))
#ax.plot.barh(df_new.T, rot=0)
ax.set_xlabel('Topic Distribution by Year')
ax.set_title('Percent of Comments in Topic For a Given Year')

st.pyplot(fig2)

# Satisfaction by topic

#df_merged.groupby(['name', 'id', 'dept'])['total_sale'].mean().reset_index()
topic_sat = [sum(survey_comments[i]*survey_comments["OverallSatisfaction"])/sum(survey_comments[i]) for i in topicnames]
topic_sat_sorted = zip(topic_sat, topicnames)
sorted_pairs = sorted(topic_sat_sorted)
tuples = zip(*sorted_pairs)
topic_sat, topicnames = [list(tuple) for tuple in tuples]

y_pos = np.arange(len(topicnames))

plt.style.use('ggplot')
fig3, ax = plt.subplots(figsize=(9,6))
hbars = ax.barh(y_pos, topic_sat)
ax.set_yticks(y_pos)
ax.set_yticklabels(topicnames)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Average Satisfaction')
ax.set_title('Average Satisfaction by Topic')
# Label with specially formatted floats
for i, v in enumerate(topic_sat):
    ax.text(v + .05, i + .2, round(v,2), color='blue', fontweight='bold')

st.pyplot(fig3)
################################################################################################

# add break line and create header for monthly analysis section
st.markdown('---')
monthly_analysis = '<p style="color:Gray; font-size: 30px;"> Month Analysis</p>'
st.markdown(monthly_analysis, unsafe_allow_html=True)

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
              'December 2020',
              'January 2021',
              'February 2021',
              'March 2021',
              'April 2021',
              'May 2021',]

month_year_option = st.selectbox('Select a month to filter the Dashboard:', month_list)
#'Month selected: ', month_year_option
print(month_year_option)
print(type(month_year_option))

# filter dataframe to satisfy the user selected criterion
date_requirement = survey_comments['month'] == month_year_option
date_filtered_survey = survey_comments[date_requirement]


## bar chart showing all-time number of negative, neutral, and positive reviews
# aggregate data by sentiment and create a bar chart
date_filtered_survey['constant'] = 1
sentiment_plot = date_filtered_survey.groupby(['sentiment']).sum()['constant'].to_frame()

fig4, ax = plt.subplots(figsize=(9,6))
sentiment_bar = ax.bar(sentiment_plot.index, sentiment_plot['constant'], color=['red', 'yellow', 'green'])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('Month Survey Volume by Sentiment')
st.pyplot(fig4)

## add horizontal bar chart by topic for user specified month
# plot pie chart of polarities
date_filtered_survey['constant'] = 1
sentiment_plot = date_filtered_survey.groupby(['sentiment']).sum()['constant'].to_frame()


sentiment_plot = date_filtered_survey.groupby(['sentiment'])["LoggingIn", "AccessPurchase", "PageNumberSearch",
                                                           "Price", "Navigation", "Products", "CustomerSupport",
                                                           "OtherDevices", "Technical", "Other(Homework)"].sum()
sentiment_plot = sentiment_plot / sentiment_plot.sum()
sentiment_plot = sentiment_plot.T.sort_values(by=['positive'])
for col in sentiment_plot.columns:
    print(col)

negative = sentiment_plot['negative'].tolist()
neutral = sentiment_plot['neutral'].tolist()
positive = sentiment_plot['positive'].tolist()
y = sentiment_plot.index.tolist()

fig5 = plt.figure(figsize=(9,6))
plt.barh(y, negative, color = 'red')
plt.barh(y, neutral, color = 'yellow', left = negative)
plt.barh(y, positive, color = 'green', left=list(map(lambda neg, neu: neg + neu, negative, neutral)))
plt.title("Month Sentiment Percentage by Topic")
#st.write("Month Sentiment Percentage by Topic")
st.pyplot(fig5)

# pie chart

# plot pie chart of polarities
date_filtered_survey['constant'] = 1
sentiment_plot = date_filtered_survey.groupby(['sentiment']).sum()['constant'].to_frame()
labels = sentiment_plot.index.to_list() # OverallSatisfaction becomes the index after the groupby
sizes = sentiment_plot['constant'].to_list()
fig6 = plt.figure(figsize=(9,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['red', 'yellow', 'green'])

pie_chart = date_filtered_survey.groupby(['Group Satisfaction']).sum()['constant'].to_frame()
labels = pie_chart.index.to_list() # OverallSatisfaction becomes the index after the groupby
sizes = pie_chart['constant'].to_list()
fig7 = plt.figure(figsize=(9,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['red', 'yellow', 'green'])

# use the two above pie charts and display them in columns
col1, col2 = st.beta_columns(2)

with col1:
    # create pie chart title and display pie chart to dashboard
    pie_chart_title = '*Proportions of Survey Sentiments*'
    st.markdown(pie_chart_title)
    st.pyplot(fig6)
    st.text("")
    st.text("")
    st.text("")
    st.text("")

with col2:
    # create pie chart title and display pie chart to dashboard
    pie_chart_title = '*Proportions of Consumer Ratings*'
    st.markdown(pie_chart_title)
    st.pyplot(fig7)
    st.text("")
    st.text("")
    st.text("")
    st.text("")

# remove unnecessary columns and display dataframe
dataframe_title = '*Data Table with Sentiment and Topic Information*'
st.markdown(dataframe_title)
survey_condensed = date_filtered_survey[['OverallSatisfaction', 'OpenResponse','sentiment']+topicnames].copy()
survey_condensed.columns = ['Consumer Satisfaction', 'Consumer Response', 'Sentiment Score', 'Access/Purchase',
                            'Customer Support', 'Technical', 'Logging in/Timing Out', 'Products', 'Other (Homework)',
                            'Price', 'Page Numbers', 'Navigation', 'Other Devices']
st.dataframe(data=survey_condensed)

st.markdown('---')
monthly_analysis = '<p style="color:Gray; font-size: 30px;"> pyLDAvis Topic Modeling Visualization</p>'
st.markdown(monthly_analysis, unsafe_allow_html=True)

HtmlFile = open("lda.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code, width = 1500, height = 1000)
