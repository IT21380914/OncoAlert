import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

#load the dataset
data = pd.read_csv("Data set 2.csv")

data.rename(columns={"Reginol Node Positive":"Regional Node Positive"}, inplace=True)
data.rename(columns={"T Stage ": "T Stage"}, inplace=True)

#remove unneccesary categorical columns
data = data.drop(['Race', 'Marital Status'], axis = 1)

data["Grade"].replace({" anaplastic; Grade IV": "4"}, inplace=True)
data["Grade"] = data["Grade"].astype(int)
data["T Stage"].replace({"T1":1, "T2": 2, "T3":3, "T4": 4}, inplace=True)
data["N Stage"].replace({"N1":1, "N2": 2, "N3":3}, inplace=True)
data["6th Stage"].replace({"IIA":1, "IIB": 2, "IIIA":3, "IIIB": 4,"IIIC":5}, inplace=True)
data["differentiate"].replace({"Moderately differentiated": 2,
                            "Poorly differentiated": 1,
                            "Well differentiated": 3,
                            "Undifferentiated": 0}, inplace=True)
data["A Stage"].replace({"Regional":1, "Distant": 0}, inplace=True)
data["Estrogen Status"].replace({"Positive":1, "Negative": 0}, inplace=True)
data["Progesterone Status"].replace({"Positive":1, "Negative": 0}, inplace=True)
data["Status"].replace({"Alive":1, "Dead": 0}, inplace=True)

df1= data.copy()

X = df1.drop(['Status'], axis = 1)
y = df1['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
rfmodel=RandomForestClassifier(n_estimators=100)
rfmodel.fit(X_train,y_train)

with open('random_forestmodel.pkl', 'wb') as file:
    pickle.dump(rfmodel, file)