import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('Data/emails.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
X = df.iloc[:,1:3001]
print(X)
Y = df.iloc[:,-1].values
print(Y)
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.25)
svc = SVC(C=1.0,kernel='rbf',gamma='auto')
svc.fit(train_x,train_y)
y_pred2 = svc.predict(test_x)
print("Accuracy Score for SVC : ", accuracy_score(y_pred2,test_y))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(knn.score(X_test, y_test))