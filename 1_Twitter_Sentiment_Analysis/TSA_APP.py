# import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# read CSV file and create independent and dependent variables
Twitter_DF = pd.read_csv(r'https://raw.githubusercontent.com/tulasinnd/Social-Metric-Insight-Projects/main/1_Twitter_Sentiment_Analysis/DATASETS/TWITTER_SENTIMENT_ANALYSIS_PROCESSED_2000.csv')

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
user_input = st.text_input("ENTER A TWEET: ")

# if the user has entered some text, make a prediction and print the result
if user_input:
    result = model.predict([user_input])[0]
    sentiment = "Positive" if result == 2 else "Negative"
    st.write(f"### PREDICTED SENTIMENT: {sentiment}")

    # print accuracy, precision, recall
    accuracy = accuracy_score(predictions, DV_test)
    precision = precision_score(predictions, DV_test, average='weighted')
    recall = recall_score(predictions, DV_test, average='weighted')
    f1 = f1_score(predictions, DV_test, average='weighted')
    st.write("### PERFORMANCE METRICS")
#     st.write(f"Accuracy: {accuracy:.2f}")
#     st.write(f"Precision: {precision:.2f}")
#     st.write(f"Recall: {recall:.2f}")
    st.write(f"### F1 SCORE: {f1:.2f}")
