import tkinter as tk
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import mysql.connector

db = mysql.connector.connect(
    host = "localhost",
    database = "titanicDB",
    user = "achintha",
    password = "5394"
)


def pred():

    # import dataset and preprocess data
    query1 = "select * from train;"
    dataset = pd.read_sql(query1, db)
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
    query2 = "select * from test;"
    query3 = "select * from gender_submission"
    testdataset = pd.read_sql(query2, db)
    actualset = pd.read_sql(query3, db)

    # fill missing values on testing dataset
    #testdataset['Age'].fillna(testdataset['Age'].mean(), inplace=True)
    #testdataset['Fare'].fillna(testdataset['Fare'].mean(), inplace=True)

    # categorical values
    testdataset = pd.get_dummies(testdataset, columns=['Sex'], drop_first=True)


    inpname = ent1.get()
    inppage = int(ent2.get())
    inppc = int(ent3.get())
    inpsex = int(ent4.get())
    inpf = int(ent5.get())
    print(inpname, inppage, inppc, inpsex, inpf)

    targets = [[inppage, inppc, inpsex, inpf]]

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

window = tk.Tk()

lbl1 = tk.Label(text="Passenger Details")
lbl1.pack()


lbl2 = tk.Label(text="Name:")
ent1 = tk.Entry()
lbl2.pack()
ent1.pack()


lbl3 = tk.Label(text="Age:")
lbl3.pack()
ent2 = tk.Entry()
ent2.pack()

lbl4 = tk.Label(text="Passenger Class:")
lbl4.pack()
ent3 = tk.Entry()
ent3.pack()

lbl5 = tk.Label(text="Sex (Male-1/Female-0):")
lbl5.pack()
ent4 = tk.Entry()
ent4.pack()

lbl6 = tk.Label(text="Fare($):")
lbl6.pack()
ent5 = tk.Entry()
ent5.pack()

btn = tk.Button(
    text="Check",
    command=pred
    )
btn.pack()

window.mainloop()
db.close()



