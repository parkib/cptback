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

url='https://drive.google.com/file/d/1_lvLY-3rlNZoOkJiCVYZIsXF2eT_swf1/view?usp=sharing'    
url='https://drive.google.com/uc?id=' + url.split('/')[-2]

class StrokesModel:
    _instance = None
    def __init__(self):
        self.model = None
        self.dt = None
        self.features = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi' ]
        self.target = 'stroke'
        self.stroke_data = pd.read_csv(url)
        #self.encoder = OneHotEncoder(handle_unknown='ignore')

        # clean the titanic dataset, prepare it for training
    def _clean(self):
        # Drop unnecessary columns
        self.stroke_data.drop(['id', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], axis=1, inplace=True)

        # Convert boolean columns to integers
        self.stroke_data['gender'] = self.stroke_data['gender'].apply(lambda x: 1 if x == 'male' else 0)
        #self.stroke_data['Residence_type'] = self.stroke_data['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
        #self.stroke_data['smoking_status'] = self.stroke_data['smoking_status'].apply(lambda x: 1 if x == 'smoked' else 0)

        
        # Drop rows with missing values
        self.stroke_data.dropna(inplace=True)

    def _train(self):
        # split the data into features and target
        X = self.stroke_data[self.features]
        y = self.stroke_data[self.target]
        
        # perform train-test split
        self.model = LogisticRegression(max_iter=1000)
        
        # train the model
        self.model.fit(X, y)
        
        # train a decision tree classifier
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)
        
    @classmethod
    def get_instance(cls):
        """ Gets, and conditionaly cleans and builds, the singleton instance of the TitanicModel.
        The model is used for analysis on titanic data and predictions on the survival of theoritical passengers.
        
        Returns:
            TitanicModel: the singleton _instance of the TitanicModel, which contains data and methods for prediction.
        """        
        # check for instance, if it doesn't exist, create it
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        # return the instance, to be used for prediction
        return cls._instance

    def predict(self, individual):
        """ Predict the survival probability of a passenger.

        Args:
            passenger (dict): A dictionary representing a passenger. The dictionary should contain the following keys:
                'pclass': The passenger's class (1, 2, or 3)
                'sex': The passenger's sex ('male' or 'female')
                'age': The passenger's age
                'sibsp': The number of siblings/spouses the passenger has aboard
                'parch': The number of parents/children the passenger has aboard
                'fare': The fare the passenger paid
                'embarked': The port at which the passenger embarked ('C', 'Q', or 'S')
                'alone': Whether the passenger is alone (True or False)

        Returns:
           dictionary : contains die and survive probabilities 
        """
        # clean the passenger data
        individual_df = pd.DataFrame(individual, index=[0])
        individual_df['gender'] = individual_df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
        #individual_df['Residence_type'] = individual_df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
        #individual_df['smoking_status'] = individual_df['smoking_status'].apply(lambda x: 1 if x == 'smoked' else 0)
        
        # predict the survival probability and extract the probabilities from numpy array
        stroke = np.squeeze(self.model.predict_proba(individual_df))
        # return the survival probabilities as a dictionary
        return {'stroke': stroke}
    
    def feature_weights(self):
        """Get the feature weights
        The weights represent the relative importance of each feature in the prediction model.

        Returns:
            dictionary: contains each feature as a key and its weight of importance as a value
        """
        # extract the feature importances from the decision tree model
        importances = self.dt.feature_importances_
        # return the feature importances as a dictionary, using dictionary comprehension
        return {feature: importance for feature, importance in zip(self.features, importances)} 
    
def initStroke():
    """ Initialize the Titanic Model.
    This function is used to load the Titanic Model into memory, and prepare it for prediction.
    """
    StrokesModel.get_instance()
    
def testStroke():
    """ Test the Titanic Model
    Using the TitanicModel class, we can predict the survival probability of a passenger.
    Print output of this test contains method documentation, passenger data, survival probability, and survival weights.
    """
     
    # setup passenger data for prediction
    #'Residence_type': ['Urban'],
    print(" Step 1:  Define theoritical passenger data for prediction: ")
    individual = {
        'gender': ['Make'],
        'age': [14],
        'hypertension': [1],
        'heart_disease': [1],
        'avg_glucose_level': [24],
        'bmi': [18],
    }
    print("\t", individual)
    print()

    # get an instance of the cleaned and trained Titanic Model
    StrokeModel = StrokesModel.get_instance()
    print(" Step 2:", StrokeModel.get_instance.__doc__)
   
    # print the survival probability
    print(" Step 3:", StrokeModel.predict.__doc__)
    probability = StrokeModel.predict(individual)
    print('\t Stroke probability: {:.2%}',(probability.get('stroke')))  
    print()
    
    # print the feature weights in the prediction model
    print(" Step 4:", StrokeModel.feature_weights.__doc__)
    importances = StrokeModel.feature_weights()
    for feature, importance in importances.items():
        print("\t\t", feature, f"{importance:.2%}") # importance of each feature, each key/value pair
        
if __name__ == "__main__":
    print(" Begin:", testStroke.__doc__)
    testStroke()



# Load the titanic dataset
## google drive link was used to allow pandas to access the csv file
url='https://drive.google.com/file/d/1_lvLY-3rlNZoOkJiCVYZIsXF2eT_swf1/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
stroke_data = pd.read_csv(url)

# Preprocess the data
## dropping the columns not necessary and relevant for the ML analysis
stroke_data.drop(['id', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], axis=1, inplace=True)

## dropping all NA values in dataset
stroke_data.dropna(inplace=True)

## convert all sex values to 0/1 (ML models can only process quantitative data)
stroke_data['gender'] = stroke_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
#stroke_data['heart_disease'] = stroke_data['heart_disease'].apply(lambda x: 1 if x == 'Yes' else 0)
#stroke_data['Residence_type'] = stroke_data['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
#stroke_data['smoking_status'] = stroke_data['smoking_status'].apply(lambda x: 1 if x == 'smoked' else 0)

# Encode categorical variables

## onehotencode was not required for this data as there weronly binary values for most variables
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
