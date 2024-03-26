from tkinter import Tk
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# import dataset and preprocess data
dataset = pd.read_csv('C:/Users/achintha.j/Documents/Titanic/train.csv')
dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)

#categorical data
dataset = pd.get_dummies(dataset, columns=['Sex'], drop_first=True)

targets = ['Pclass', 'Sex_male', 'Age', 'Fare']
X = dataset[targets]
y = dataset['Survived']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#base models
model1 = RandomForestClassifier(random_state=42)
model2 = GradientBoostingClassifier(random_state=42)
model3 = LogisticRegression(random_state=42)

ensemble = VotingClassifier(estimators=[('rf', model1), ('gb', model2), ('lr', model3)], voting='soft')
ensemble.fit(X_train, y_train)

#predictions on test_data
predictions = ensemble.predict(X_test)

print(predictions)

# Load test dataset to the model
testdataset = pd.read_csv('C:/Users/achintha.j/Documents/Titanic/test.csv')
actualset = pd.read_csv('C:/Users/achintha.j/Documents/Titanic/gender_submission.csv')

# fill missing values on testing dataset
#testdataset['Age'].fillna(testdataset['Age'].mean(), inplace=True)
#testdataset['Fare'].fillna(testdataset['Fare'].mean(), inplace=True)

# categorical values
testdataset = pd.get_dummies(testdataset, columns=['Sex'], drop_first=True)


pclass = int(input('Passenger Class: '))
sex = int(input('Sex (Male-1, Female-0): '))
age = int(input('Age: '))
fare = int(input('Fare: '))

targets = [[pclass, sex, age, fare]]

X = targets

#get predictions on the testing dataset
test_predictions = ensemble.predict(X)

print(test_predictions)

if (test_predictions == 0):
    print('Not Survived')
elif (test_predictions == 1):
    print('Survived')
else:
    print('Error')

5