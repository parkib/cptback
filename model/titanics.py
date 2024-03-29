import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class TitanicModel:
    """A class used to represent the Titanic Model for passenger survival prediction."""

    _instance = None

    def __init__(self):
        self.model = None
        self.dt = None
        self.features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'alone']
        self.target = 'survived'
        self.titanic_data = None
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def _clean_data(self):
        """Clean the titanic dataset."""
        self.titanic_data = sns.load_dataset('titanic')
        self.titanic_data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
        self.titanic_data['sex'] = self.titanic_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
        self.titanic_data['alone'] = self.titanic_data['alone'].apply(lambda x: 1 if x == True else 0)
        self.titanic_data.dropna(subset=['embarked'], inplace=True)
        
        onehot = self.encoder.fit_transform(self.titanic_data[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.titanic_data = pd.concat([self.titanic_data, onehot_df], axis=1)
        self.features.extend(cols)

    def _train(self):
        """Train the Titanic model."""
        X = self.titanic_data[self.features]
        y = self.titanic_data[self.target]
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)

    @classmethod
    def get_instance(cls):
        """Gets the singleton instance of the TitanicModel."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean_data()
            cls._instance._train()
        return cls._instance

    def predict(self, passenger):
        """Predict the survival probability of a passenger."""
        passenger_df = pd.DataFrame(passenger, index=[0])
        passenger_df['sex'] = passenger_df['sex'].apply(lambda x: 1 if x == 'male' else 0)
        passenger_df['alone'] = passenger_df['alone'].apply(lambda x: 1 if x == True else 0)
        onehot = self.encoder.transform(passenger_df[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        passenger_df = pd.concat([passenger_df, onehot_df], axis=1)
        passenger_df.drop(['embarked', 'name'], axis=1, inplace=True)
        
        die, survive = np.squeeze(self.model.predict_proba(passenger_df))
        return {'die': die, 'survive': survive}

    def feature_weights(self):
        """Get the feature weights."""
        importances = self.dt.feature_importances_
        return {feature: importance for feature, importance in zip(self.features, importances)} 


def initTitanic():
    """Initialize the Titanic Model."""
    TitanicModel.get_instance()
    
def testTitanic():
    """Test the Titanic Model."""
    print("Defining theoretical passenger data for prediction:")
    passenger = {
        'name': ['John Mortensen'],
        'pclass': [2],
        'sex': ['male'],
        'age': [64],
        'sibsp': [1],
        'parch': [1],
        'fare': [16.00],
        'embarked': ['S'],
        'alone': [False]
    }
    print("\t", passenger)
    print()

    titanicModel = TitanicModel.get_instance()
    print("Step 2:", titanicModel.get_instance.__doc__)
    
    print("Step 3:", titanicModel.predict.__doc__)
    probability = titanicModel.predict(passenger)
    print('\tDeath probability: {:.2%}'.format(probability.get('die')))  
    print('\tSurvival probability: {:.2%}'.format(probability.get('survive')))
    print()
    
    print("Step 4:", titanicModel.feature_weights.__doc__)
    importances = titanicModel.feature_weights()
    for feature, importance in importances.items():
        print("\t", feature, f"{importance:.2%}")

if __name__ == "__main__":
    print("Begin:", testTitanic.__doc__)
    testTitanic()

# Loading the Titanic dataset
titanic_data = sns.load_dataset('titanic')
titanic_data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
titanic_data.dropna(inplace=True)
titanic_data['sex'] = titanic_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
titanic_data['alone'] = titanic_data['alone'].apply(lambda x: 1 if x == True else 0)

# Encoding categorical variables
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(titanic_data[['embarked']])
onehot = enc.transform(titanic_data[['embarked']]).toarray()
cols = ['embarked_' + val for val in enc.categories_[0]]
titanic_data[cols] = pd.DataFrame(onehot)
titanic_data.drop(['embarked'], axis=1, inplace=True)
titanic_data.dropna(inplace=True)

# Splitting the data into training and testing sets
X = titanic_data.drop('survived', axis=1)
y = titanic_data['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training a decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Testing the model
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
