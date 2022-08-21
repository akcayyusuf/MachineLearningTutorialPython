import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import linear_model

df = pd.read_csv("homeprices.csv")


medianBedroom=math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(medianBedroom)

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

print(reg.predict([[3000,3,40]]))

