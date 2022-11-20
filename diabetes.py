import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/diabetes.csv')
print(df)
print(df.head())
print(df.info())
print(df.isnull().sum())

X = df.iloc[:8]
Y = df.iloc[:8]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_fit = knn.fit(X_train, Y_train)
knn_pred = knn_fit.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

print("confusion_matrix")
print(confusion_matrix(Y_test, knn_pred))
print("Accuracy Score:", accuracy_score(Y_test, knn_pred))
print("Recal Score :", recall_score(Y_test, knn_pred))
print("F1 Score :", f1_score(Y_test, knn_pred))
print("Precision Score :", precision_score(Y_test,knn_pred))