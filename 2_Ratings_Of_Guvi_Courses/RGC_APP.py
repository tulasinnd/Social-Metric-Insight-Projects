import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import streamlit as st

# Load the dataset
Guvi_DF_CLEAN = pd.read_csv('RATINGS_OF_GUVI_COURSES_CLEANED_DATA.csv')

# Split data into features and target
X = Guvi_DF_CLEAN[['price', 'num_subscribers', 'num_reviews', 'num_lectures', 'content_duration', 'level', 'subject']]
y = Guvi_DF_CLEAN['Rating']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline to scale and reduce dimensionality using PCA
pipe = make_pipeline(StandardScaler(), PCA(n_components=4), RandomForestRegressor(n_estimators=100, random_state=42))

# Train the model
pipe.fit(X_train, y_train)

# Make predictions on test data
y_pred = pipe.predict(X_test)

# Evaluate the model performance
print('R-squared score:', r2_score(y_test, y_pred))

# Streamlit app
st.title('GUVI Rating Prediction')
st.write('Enter the course details to predict its rating:')

# Get user input
price = st.slider('Price', min_value=0, max_value=200, value=100, step=1)
num_subscribers = st.slider('Number of Subscribers', min_value=0, max_value=100000, value=5000, step=10)
num_reviews = st.slider('Number of Reviews', min_value=0, max_value=1000, value=100, step=1)
num_lectures = st.slider('Number of Lectures', min_value=0, max_value=500, value=50, step=1)
content_duration = st.slider('Content Duration (in minutes)', min_value=0, max_value=1000, value=100, step=1)
level = st.selectbox('Course Level', ['All Levels', 'Beginner Level', 'Intermediate Level', 'Expert Level'])
subject = st.selectbox('Course Subject', ['Business Finance', 'Graphic Design', 'Musical Instruments', 'Web Development'])

# Create a DataFrame for user input
input_df = pd.DataFrame({
    'price': price,
    'num_subscribers': num_subscribers,
    'num_reviews': num_reviews,
    'num_lectures': num_lectures,
    'content_duration': content_duration,
    'level': level,
    'subject': subject
}, index=[0])

# Transform user input using the same pipeline
input_transformed = pipe.transform(input_df)

# Make prediction using transformed input
prediction = pipe.predict(input_transformed)[0]

# Display predicted rating
st.write('Predicted Rating:', prediction)
