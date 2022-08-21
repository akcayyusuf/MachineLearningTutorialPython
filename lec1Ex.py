import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

df = pd.read_csv('canada_per_capita_income.csv')

plt.xlabel("Years")
plt.ylabel("Income")

plt.scatter(df.year,df.income,color='red',marker='+')

reg=linear_model.LinearRegression()
reg.fit(df[["year"]].values,df["income"].values)

plt.plot(df.year,reg.predict(df[["year"]]),color='blue' )
print(reg.predict([[2020]]))
plt.scatter([[2020]],reg.predict([[2020]]),color='g',marker='^')
plt.show()