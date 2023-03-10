import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
# import streamlit as st

# Load the dataset
Guvi_DF_CLEAN = pd.read_csv('2_Ratings_Of_Guvi_Courses/RATINGS_OF_GUVI_COURSES_CLEANED_DATA.csv')

# # Split data into features and target
# X = Guvi_DF_CLEAN[['price', 'num_subscribers', 'num_reviews', 'num_lectures', 'content_duration', 'level', 'subject']]
# y = Guvi_DF_CLEAN['Rating']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a pipeline to scale and reduce dimensionality using PCA
# pipe = make_pipeline(StandardScaler(), PCA(n_components=4), RandomForestRegressor(n_estimators=100, random_state=42))

# # Train the model
# pipe.fit(X_train, y_train)

# # Make predictions on test data
# y_pred = pipe.predict(X_test)

# # Evaluate the model performance
# print('R-squared score:', r2_score(y_test, y_pred))

# # Streamlit app
# st.title('GUVI Rating Prediction')
# st.write('Enter the course details to predict its rating:')

# # Get user input
# price = st.slider('Price', min_value=0, max_value=200, value=100, step=1)
# num_subscribers = st.slider('Number of Subscribers', min_value=0, max_value=100000, value=5000, step=10)
# num_reviews = st.slider('Number of Reviews', min_value=0, max_value=1000, value=100, step=1)
# num_lectures = st.slider('Number of Lectures', min_value=0, max_value=500, value=50, step=1)
# content_duration = st.slider('Content Duration (in minutes)', min_value=0, max_value=1000, value=100, step=1)
# level = st.selectbox('Course Level', ['All Levels', 'Beginner Level', 'Intermediate Level', 'Expert Level'])
# subject = st.selectbox('Course Subject', ['Business Finance', 'Graphic Design', 'Musical Instruments', 'Web Development'])

# # Create a DataFrame for user input
# input_df = pd.DataFrame({
#     'price': price,
#     'num_subscribers': num_subscribers,
#     'num_reviews': num_reviews,
#     'num_lectures': num_lectures,
#     'content_duration': content_duration,
#     'level': level,
#     'subject': subject
# }, index=[0])

# # Transform user input using the same pipeline
# input_transformed = pipe.transform(input_df)

# # Make prediction using transformed input
# prediction = pipe.predict(input_transformed)[0]

# # Display predicted rating
# st.write('Predicted Rating:', prediction)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Drop irrelevant columns
Guvi_DF_CLEAN = Guvi_DF_CLEAN.drop(['course_id', 'course_title', 'url', 'published_timestamp'], axis=1)

# Handle missing values
Guvi_DF_CLEAN = Guvi_DF_CLEAN.dropna()

# Encode categorical variables
Guvi_DF_CLEAN = pd.get_dummies(Guvi_DF_CLEAN, columns=['level', 'subject'])

# Split data into training and testing sets
X = Guvi_DF_CLEAN.drop(['Rating'], axis=1)
y = Guvi_DF_CLEAN['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for multicollinearity
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif["features"] = X_train.columns
print(vif)

# Remove predictors with high VIF values
X_train = X_train.drop(['num_subscribers', 'subject_Web Development'], axis=1)
X_test = X_test.drop(['num_subscribers', 'subject_Web Development'], axis=1)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate the model performance
print('R-squared score:', r2_score(y_test, y_pred))

# Streamlit app
import streamlit as st

st.title("GUVI ML Regression Algorithm using Random Forest")
st.write("Please enter the following details to predict the rating:")

# Get user input
price = st.number_input("Price")
num_reviews = st.number_input("Number of reviews")
num_lectures = st.number_input("Number of lectures")
content_duration = st.number_input("Content duration")
level = st.selectbox("Level", ['All Levels', 'Beginner Level', 'Intermediate Level', 'Expert Level'])
subject = st.selectbox("Subject", ['Business Finance', 'Graphic Design', 'Musical Instruments', 'Web Development'])

# Encode user input
level_dummies = pd.get_dummies([level])[0].reindex(columns=X_train.columns, fill_value=0)
subject_dummies = pd.get_dummies([subject])[0].reindex(columns=X_train.columns, fill_value=0)
user_input = pd.DataFrame({'price': [price],
                           'num_reviews': [num_reviews],
                           'num_lectures': [num_lectures],
                           'content_duration': [content_duration]})
user_input = pd.concat([user_input, level_dummies, subject_dummies], axis=1)

# Make prediction on user input
prediction = model.predict(user_input)[0]

# Display prediction
st.write(f"Predicted rating: {prediction:.2f}")

