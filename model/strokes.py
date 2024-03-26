import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import sqlite3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import seaborn as sns

# Load the titanic dataset
## google drive link was used to allow pandas to access the csv file
url='https://drive.google.com/file/d/1_lvLY-3rlNZoOkJiCVYZIsXF2eT_swf1/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
stroke_data = pd.read_csv(url)

# Preprocess the data
## dropping the columns not necessary and relevant for the ML analysis
stroke_data.drop(['id', 'ever_married', 'work_type'], axis=1, inplace=True)

## dropping all NA values in dataset
stroke_data.dropna(inplace=True)

## convert all sex values to 0/1 (ML models can only process quantitative data)
stroke_data['gender'] = stroke_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
#stroke_data['heart_disease'] = stroke_data['heart_disease'].apply(lambda x: 1 if x == 'Yes' else 0)
stroke_data['Residence_type'] = stroke_data['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
stroke_data['smoking_status'] = stroke_data['smoking_status'].apply(lambda x: 1 if x == 'smoked' else 0)

# Encode categorical variables

## onehotencode was not required for this data as there were only binary values for most variables
## enc = OneHotEncoder(handle_unknown='ignore')
## enc.fit(stroke_data[['embarked']])
## onehot = enc.transform(titanic_data[['embarked']]).toarray()
## cols = ['embarked_' + val for val in enc.categories_[0]]
## titanic_data[cols] = pd.DataFrame(onehot)
## titanic_data.drop(['embarked'], axis=1, inplace=True)
##titanic_data.dropna(inplace=True)

# Split the data into training and testing sets
X = stroke_data.drop('stroke', axis=1)
y = stroke_data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
#dt = DecisionTreeClassifier()
#dt.fit(X_train, y_train)

# Test the model
#y_pred = dt.predict(X_test)

## slightly lower accuracies
# X = stroke_data.drop('stroke', axis=1)
# y = stroke_data['stroke']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
#dt = DecisionTreeClassifier()
#dt.fit(X_train, y_train)
# Test the model
#y_pred = dt.predict(X_test)

## gaussian naive bayes - a classification technique that can also be used for regression
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
## accuracy was approximatey 89%
print('Accuracy:', accuracy)
