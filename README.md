# SOCIAL METRIC INSIGHT PROJECTS

    1. Twitter Sentiment Analysis Application Link: https://tulasinnd-social-met-1-twitter-sentiment-analysistsa-app-lr5buv.streamlit.app/
    2. 
    3. 

# 1 Twitter Sentiment Analysis:

INTRODUCTION:

    The goal of this project is to create a model that can classify tweets as positive, negative, 
    or neutral based on their content.This can be useful for sentiment analysis, brand monitoring, 
    and other applications where it's important to understand how people feel about a particular 
    topic or product. In order to accomplish this task, we need to preprocess the data, select 
    appropriate features, and train a machine learning model to make accurate predictions.

DATA PRE-PROCESSING:

    1.	The dataset consists of six columns, with the target feature being the dependent variable.
    2.	The dataset does not have any null or missing values.
    3.	Features such as IDs, date, flag, and user are not relevant for sentiment analysis and
        have been removed.
    4.	Noise dpunctuationpunctuations, special characters, and URLs have been removed.
    5.	The text has been converted to lowercase and stopwords have been removed.
    6.	As a result of these pre-processing steps, the target column now consists of three values, 
        0 for negative, and 2 for positive.
    7.	The target column has an equal number of positive and negative values, making it a 
        balanced dataset

SPLIT THE DATASET INTO TRAIN TEST SPLIT:

    1.	Randomly split the dataset into training and testing sets, with 80% for training 
        and 20% for testing.

FEATURE EXTRACTION:

    1.	Convert the text data into numerical features that can be used by the machine learning algorithms.
    2.	Use TF-IDF to extract features from the text data.

TRAIN CLASSIFICATION MODEL:

    1.	Use logistic regression Algorithm to train the classification model.
    2.	Train the model on the training set and evaluate its performance on the testing set.

EVALUATE MODEL:

    1.	Calculate the accuracy, precision, recall, and F1-score of the model on the testing set.

DEPLOYMENT:

    1.	Once you have a model that performs well on the testing set, you can deploy it to predict the 
         sentiment of new tweets in real-time.
    2.	Here I have used streamlit cloud.

# 2 RATINGS OF GUVI COURSES
# 3 INSTAGRAM INFLUENCERS
