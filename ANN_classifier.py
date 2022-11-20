import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv('Data/Churn_Modelling.csv', index_col = 'RowNumber')
print(df.head())
print(df.info())
print(df.isnull().sum())

X_columns = df.columns.tolist()[2:12]
Y_columns = df.columns.tolist()[-1:]
print(X_columns)
print(Y_columns)

X = df[X_columns].values 
Y = df[Y_columns].values

from sklearn.preprocessing import LabelEncoder
X_column_transformer = LabelEncoder()
X[:, 1] = X_column_transformer.fit_transform(X[:, 1])

X[:, 2] = X_column_transformer.fit_transform(X[:, 2])

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
			[
				('Categorizer', ColumnTransformer(
						[
							("Gender Label Encoder", OneHotEncoder(categories = 'auto', drop = 'first'), [2]),
							("Geography Label Encoder", OneHotEncoder(categories = 'auto', drop = 'first'), [1])
						],
						remainder = 'passthrough', n_jobs = 1)),
					('Normalizer', StandardScaler())
			]
)

X = pipeline.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 )

from keras.model import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Dense(6, activation = 'relu', input_shape = (X_train.shape[1], )))