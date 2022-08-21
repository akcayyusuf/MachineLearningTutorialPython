import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

df = pd.read_csv("homeprices.csv")

plt.xlabel("area")
plt.ylabel("Price")
plt.scatter(df.area,df.price,color='red',marker='+')
reg =linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

plt.plot(df.area,reg.predict(df[['area']]),color='blue' )


plt.show()

