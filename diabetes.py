import pandas as pd
import numpy as np
from numpy import nan

x='./diabetes.csv'
df=pd.read_csv(x)

# To summarize the dataset
print(df.describe())

print(df.head)

# To list the columns present in the dataset
#print(df.columns)

# To #print the data types of the data set
#print(df.dtypes)


#print(df.columns)

# To list the number of missing values
drop_values=(df[['BloodPressure', 'Glucose', 'Insulin', 'BMI', 'SkinThickness']]==0).sum()

print(drop_values)

# Set the replaced data into the data frame

cols = ["BloodPressure", "Glucose", "Insulin", "BMI", "SkinThickness"]
df[cols] = df[cols].replace(0, nan)

# Drop the values of nan
df.dropna(subset=['BloodPressure', 'Glucose', 'Insulin', 'BMI', 'SkinThickness'], axis=0, inplace=True)

print(df.shape)

print(df.head)


