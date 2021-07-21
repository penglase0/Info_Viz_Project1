# Authors: Courtney Datin and Oliver Penglase
# Course: DS5500 - Information Visualization in Data Science
# Project Phase 2 - Classification of Student Response Surveys

# import necessary packages
import pandas as pd
import numpy as np

# read in csv dataset as a dataframe
survey_comments = pd.read_csv('Phase2Dataset.csv')

pos_survey_comments = survey_comments[survey_comments['positive'] == 1].reset_index(drop=True)
unavailable_survey_comments = survey_comments[survey_comments['positive'] != 1].reset_index(drop=True)


#def categorization(df):
    #if df['rating']  < 3:

# create a list of the conditions for the rating column (negative, neutral, positive)
conditions = [
    (unavailable_survey_comments['rating'] <= 3),
    (unavailable_survey_comments['rating'] > 3) & (unavailable_survey_comments['rating'] <=7),
    (unavailable_survey_comments['rating'] >7)
]

# create a list of possible values for the rating conditions (negative = -1, neutral = 0, positive = 1)
assigned_values = [-1, 0, 1]

# create a new column for the values
unavailable_survey_comments['adjusted_rating'] = np.select(conditions, assigned_values)

# write dataframe with adjusted rating column to csv
pos_survey_comments['adjusted_rating'] = pos_survey_comments['positive']
combined_survey = pos_survey_comments.append(unavailable_survey_comments)
combined_survey.to_csv('Phase2Dataset_v2.csv')


