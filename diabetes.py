import pandas as pd
import numpy as np
from numpy import nan
import matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


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

