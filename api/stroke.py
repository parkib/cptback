from flask import Blueprint, jsonify, Flask, request
from flask_restful import Api, Resource
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sqlite3
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

# Create a Flask application
app = Flask(__name__)

# Define a blueprint for the stroke prediction API
stroke_api = Blueprint('stroke_api', __name__, url_prefix='/api/stroke')
api = Api(stroke_api)

class Predict(Resource):
    """A class representing the endpoint for predicting stroke probability."""
    
    def post(self):
        """Handle POST requests."""
        try:
            # Get the JSON data from the request
            data = request.get_json()
            # Create a DataFrame from the JSON data
            stroke_data = pd.DataFrame(data, index=[0])
            stroke_data.head(5)
            
            # Preprocess the data
            stroke_data.dropna(inplace=True)
            stroke_data['gender'] = stroke_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
            stroke_data['Residence_type'] = stroke_data['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
            stroke_data['smoking_status'] = stroke_data['smoking_status'].apply(lambda x: 1 if x == 'smoked' else 0)
            
            # Predict the stroke probability for the new data using a Gaussian Naive Bayes model
            stroke_prob = gnb.predict_proba(stroke_data)[:, 1]
            
            # Return the predicted stroke probability as a percentage
            return {'Chance of Stroke': float(stroke_prob * 100)}, 200
        except Exception as e:
            # Return an error message if an exception occurs
            return {'error': str(e)}, 400

# Add the Predict resource to the stroke_api with the '/predict' endpoint
api.add_resource(Predict, '/predict')

# Register the stroke_api blueprint with the Flask application
app.register_blueprint(stroke_api)

# Load the stroke dataset from Google Drive
url = 'https://drive.google.com/file/d/1_lvLY-3rlNZoOkJiCVYZIsXF2eT_swf1/view?usp=sharing'
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
stroke_data = pd.read_csv(url)

# Preprocess the data

# Drop the columns 'id', 'ever_married', and 'work_type' from the stroke_data DataFrame
# These columns are dropped because they are not relevant for predicting stroke probability
stroke_data.drop(['id', 'ever_married', 'work_type'], axis=1, inplace=True)

# Remove rows with missing values (NaN) from the stroke_data DataFrame
# Missing values can affect the accuracy of the predictive model, so it's better to drop them
stroke_data.dropna(inplace=True)

# Convert the 'gender' column values to binary (0 for female, 1 for male)
# This transformation is necessary because machine learning models require numerical input
stroke_data['gender'] = stroke_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Convert the 'Residence_type' column values to binary (0 for Rural, 1 for Urban)
# This transformation is necessary because machine learning models require numerical input
stroke_data['Residence_type'] = stroke_data['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)

# Convert the 'smoking_status' column values to binary (0 for non-smoker, 1 for smoker)
# This transformation is necessary because machine learning models require numerical input
stroke_data['smoking_status'] = stroke_data['smoking_status'].apply(lambda x: 1 if x == 'smoked' else 0)


# Split the data into features and target
X = stroke_data.drop('stroke', axis=1)
y = stroke_data['stroke']

## Gaussian naive bayes was tested instead of the original model and it ended up having a slightly lower accuracy
gnb = GaussianNB()
gnb.fit(X, y)