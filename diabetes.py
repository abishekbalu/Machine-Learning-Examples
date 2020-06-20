import pandas as pd
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
plt.style.use('seaborn-whitegrid')


x='./diabetes.csv'
df=pd.read_csv(x)

# To summarize the dataset
# print(df.describe())

# print(df.head)

# To list the columns present in the dataset
#print(df.columns)

# To #print the data types of the data set
#print(df.dtypes)


# print(df.columns)


# To list the number of missing values
drop_values=(df[['BloodPressure', 'Glucose', 'Insulin', 'BMI', 'SkinThickness']]==0).sum()

# print(drop_values)

# Set the replaced data into the data frame

cols = ["BloodPressure", "Glucose", "Insulin", "BMI", "SkinThickness"]
df[cols] = df[cols].replace(0, nan) \



# Drop the values of nan
df.dropna(subset=['BloodPressure', 'Glucose', 'Insulin', 'BMI', 'SkinThickness'], axis=0, inplace=True)



print(df.shape)

print(df.head)

# Normalize the data

df[cols] = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())

Normalized_cols = df[cols]

print(df.head())

#corelation

print(df.corr())

X = df['Insulin'].values.reshape(-1, 1)

y = df.Glucose

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


#defining the model

data_model = RandomForestRegressor()

#Fit the model

data_model.fit(train_X, train_y)

# Predict the model

preds_Data = data_model.predict(val_X)

print(preds_Data)

plt.scatter(X, y)
plt.show()

print(mean_absolute_error(val_y, preds_Data))

accuracy = data_model.score(val_y, preds_Data)

print(accuracy)
