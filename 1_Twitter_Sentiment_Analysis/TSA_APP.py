# import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# read CSV file and create independent and dependent variables
# DATASETS
Twitter_DF1 = pd.read_csv(r'DATASETS/TWITTER_SENTIMENT_ANALYSIS_PROCESSED1.csv')
Twitter_DF2 = pd.read_csv(r'DATASETS/TWITTER_SENTIMENT_ANALYSIS_PROCESSED2.csv')
Twitter_DF3 = pd.read_csv(r'DATASETS/TWITTER_SENTIMENT_ANALYSIS_PROCESSED3.csv')
Twitter_DF4 = pd.read_csv(r'DATASETS/TWITTER_SENTIMENT_ANALYSIS_PROCESSED4.csv')
Twitter_DF = pd.concat([Twitter_DF1,Twitter_DF2,Twitter_DF3,Twitter_DF4], axis=0)

# recognise independent and dependent features
Independent_var = Twitter_DF['text'] 
Dependent_var = Twitter_DF['final_target']

# split data into training and testing sets
IV_train, IV_test, DV_train, DV_test = train_test_split(Independent_var, Dependent_var, test_size=0.1, random_state=225)

# create a TfidfVectorizer object and a LogisticRegression object
tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver="lbfgs", max_iter=200000)

# create a pipeline with the vectorizer and classifier objects
model = Pipeline([('vectorizer',tvec),('classifier',clf2)])

# fit the model on the training set
model.fit(IV_train, DV_train)

# make predictions on the test set
predictions = model.predict(IV_test)

# create a Streamlit app
st.title("Twitter Sentiment Analysis")

# create a text input for the user to enter a tweet
user_input = st.text_input("Enter a tweet:")

# if the user has entered some text, make a prediction and print the result
if user_input:
    result = model.predict([user_input])[0]
    sentiment = "Positive" if result == 2 else "Negative"
    st.write(f"Predicted sentiment: {sentiment}")

    # print accuracy, precision, recall, and confusion matrix
    accuracy = accuracy_score(predictions, DV_test)
    precision = precision_score(predictions, DV_test, average='weighted')
    recall = recall_score(predictions, DV_test, average='weighted')
    cm = confusion_matrix(predictions, DV_test)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write("Confusion matrix:")
    st.write(cm)
# pandas==1.3.3
# scikit-learn==1.1.1
# streamlit==1.1.0
