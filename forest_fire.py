import pandas as pd
import numpy as np
# from sklearn import DecisionTreeRegressor


x = './amazon.csv'

df = pd.read_csv(x, encoding='latin1')

print(df.describe())

print(df.head)

print(df.columns)

