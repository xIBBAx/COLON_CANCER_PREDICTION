import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet

label_encoder = preprocessing.LabelEncoder()

df = pd.read_csv("be/utils/colo_new_repeat.csv")
df['GROUP'] = label_encoder.fit_transform(df['Group'])
df['SEX'] = df['SEX'].replace({'M': 0, 'F': 1})
df.drop(["Group"], axis=1)
output_features = ['DV 1st day', '1', '2', '4', '8', '12', 'DV 2nd day', 'DV 3rd day',
                   'DV 4th day', 'DV 7th day', '145', '146', '148', '150', '152', '156', '167']
input_features = ['RATE', 'WEIGHT', 'AGE', 'SEX', 'CRCL']

X = df[input_features]
y = df[output_features]
# X = df.drop(["1", "2"], axis=1)
# y1 = df["1"]
# y2 = df["2"]


X_train, X_test, y1_train, y1_test = train_test_split(
    X, y, train_size=0.7, shuffle=True, random_state=1)
# _, _, y2_train, y2_test = train_test_split(X, y2, train_size=0.7, shuffle=True, random_state=1)


model1 = Ridge(alpha=0.1)
model1.fit(X_train, y1_train)
# models_dict['model1'] = model1

# model2 = LinearRegression()
# model2.fit(X_train, y2_train)
# models_dict['model2'] = model2

# Save the trained models
with open('model_ridge.pkl', 'wb') as file:
    pickle.dump(model1, file)
