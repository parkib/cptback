import seaborn as sns  # Seaborn is a Python visualization library based on matplotlib, used for statistical plotting
import pandas as pd  # Pandas is a data manipulation and analysis library in Python
from sklearn.model_selection import train_test_split  # This function splits arrays or matrices into random train and test subsets
from sklearn.tree import DecisionTreeClassifier  # DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset
from sklearn.metrics import accuracy_score  # Accuracy_score is a function to measure the accuracy of classification models
from sklearn.preprocessing import OneHotEncoder  # OneHotEncoder converts categorical integer features into binary numerical features
import sqlite3  # SQLite is a C library that provides a lightweight disk-based database
from sklearn.datasets import load_iris  # Load_iris is a function to load the iris dataset for classification
from sklearn.naive_bayes import GaussianNB  # GaussianNB is a class for Gaussian Naive Bayes classification

# Load the stroke dataset
## Google Drive link was used to allow Pandas to access the CSV file
url = 'https://drive.google.com/file/d/1_lvLY-3rlNZoOkJiCVYZIsXF2eT_swf1/view?usp=sharing'
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
stroke_data = pd.read_csv(url)

# Preprocess the data
## Dropping the columns not necessary and relevant for the ML analysis
stroke_data.drop(['id', 'ever_married', 'work_type'], axis=1, inplace=True)

## Dropping all NA values in the dataset
stroke_data.dropna(inplace=True)

## Convert all gender values to 0/1 (ML models can only process quantitative data)
stroke_data['gender'] = stroke_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
stroke_data['Residence_type'] = stroke_data['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
stroke_data['smoking_status'] = stroke_data['smoking_status'].apply(lambda x: 1 if x == 'smoked' else 0)

# Split the data into features (X) and target variable (y)
X = stroke_data.drop('stroke', axis=1)  # Features
y = stroke_data['stroke']  # Target variable

# Split the data into training and testing sets
## The dataset is divided into training and testing sets to evaluate the model's performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Gaussian Naive Bayes classifier
## Gaussian Naive Bayes is a classification algorithm based on Bayes' theorem with the assumption of independence between features
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

# Evaluate the model's accuracy
## The accuracy score measures the proportion of correctly classified instances out of all instances
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
## The accuracy of the trained model is printed to evaluate its performance
print('Accuracy:', accuracy)