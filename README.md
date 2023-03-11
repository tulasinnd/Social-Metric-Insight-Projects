# SOCIAL METRIC INSIGHT PROJECTS
    This repository features a range of projects related to social media. The first project, 
    'Twitter Sentiment Analysis', is designed to assist users in determining whether a given 
    tweet is positive or negative. This can be particularly useful for businesses or individuals
    who want to monitor public sentiment about their brand or a particular topic.

    The second project, 'Ratings of Guvi Courses', is focused on helping the Guvi company 
    predict the ratings that their courses will receive from learners. By doing so, the company
    can identify areas where improvements are needed to enhance the overall learning experience
    for their learners.

    Lastly, the 'Instagram Influencers' project consists of a series of questions that can help
    improve various data processing techniques.

    Overall, this repository provides a valuable resource for anyone interested in exploring 
    the intersection of social media and data analytics. Each project offers unique insights 
    and tools for analyzing and optimizing your social media strategy.

Twitter Sentiment Analysis Application Link: 
https://tulasinnd-social-met-1-twitter-sentiment-analysistsa-app-lr5buv.streamlit.app/


Ratings of Guvi Courses Application Link:
https://tulasinnd-social-metric-2-ratings-of-guvi-coursesrgc-app-wey5zi.streamlit.app/


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

# 2 Ratings of Guvi Courses

INTRODUCTION:
    
        This model to predicts the ratings given by the learners to the course based 
        on various factors like price, number of suscribers, number of reviews, 
        course subject, content duration

DATA CLEANING:

        Load the dataset
            Dataset contains 3680 rows and 11 columns
            
        Handling missing values
            Dataset contains negligible(< 0.1%) valus so deleting them don't effect
            the dataset and no loss of information
            
        Checking for duplicates
            No duplicate values present 
            
        Removing irrelevant columns
            I have deleted irrelevent columns that do not effect target column are 
            course_id, course_title,url,published_timestamp
            
        Plottings
            Plottings between different features are plotted to find the nature 
            of columns, outliers and correlation
            
        Encoding categorical variables
            Categorical data is encoded since ML works with numbers, here I 
            have used label encoder
            
        Handling outliers
            There are outliers present in the data that are detected using various
            plots like box plot,histogram and scatter plot, more than 10 % data contains
            outliers, inorder to handle these outliers I have used log tansformation
            
        Multicolliniarity
            After finishing the entire dataset I have checked for multicolliniarity, 
            found that the vif score of the columns is very high(10 to 20) that 
            need to be handled while model tuning
            
MODEL TRAINING, TESTING EVALUATING AND TRY NEW VALUES:
  
Importing the necessary libraries

Splitting the data into training and testing sets:

    Train_test_split() function is used to randomly split the data into training 
    and testing sets.

    'test_size' parameter specifies the proportion of the data to be used for testing.

    'random_state' parameter is used for reproducibility of the results.

Defining the preprocessing steps for numerical and categorical features:

    Two separate pipelines are defined for numerical and categorical features.
    num_transformer pipeline has two steps:
    i. SimpleImputer with 'median' strategy to fill missing values in numerical 
    features with the median of the column.
    ii. StandardScaler to standardize the values of numerical features.

    cat_transformer pipeline also has two steps:
    i. SimpleImputer with 'most_frequent' strategy to fill missing values in 
    categorical features with 
    the most frequent value in the column.
    ii. OneHotEncoder to one-hot encode the categorical features.

Combining the preprocessing steps using ColumnTransformer:

    a. ColumnTransformer combines the num_transformer and cat_transformer pipelines.
    b. transformers parameter specifies the pipeline for each type
    of feature (numerical or categorical).
    c. The 'transformers' parameter takes a list of tuples. Each tuple 
    specifies the name of the pipeline, the pipeline itself, and 
    the names of the columns to be transformed.

Defining the Random Forest regression model:

    a. RandomForestRegressor is defined with 100 decision trees and random state 42.

Creating the pipeline:

    a. Pipeline is created with the 'preprocessor' and 'regressor' steps.
    b. 'preprocessor' is the ColumnTransformer object that preprocesses the data
    before feeding it to the model.
    c. 'regressor' is the RandomForestRegressor object that predicts the target variable.

Fitting the pipeline on the training data:

    a. fit() method is called on the pipeline object to train the model on the training data.

Making predictions on the test data:

    a. predict() method is called on the pipeline object to make predictions on the test data.

Evaluating the model performance:

    a. r2_score() function is used to evaluate the model performance

# 3 Instagram Influencers

    Load the instagram influencers dataset and answer the given five questions, 
    these questions will help us understand various data processing techniques
        1 Dealing with various datatypes and type conversion of features
        2 Find the correlation between the columns
        3 Finding the data distributions of columns
        4 Understanding aggregate functions of dataframes
        5 Understanding the relation between columns
