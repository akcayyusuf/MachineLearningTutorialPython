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

import pickle
with open('model_pickle','wb') as f:
    pickle.dump(reg,f)

with open('model_pickle','rb') as f:
    savedModel=pickle.load(f)
print(reg.predict([[5000]]))
print(savedModel.predict([[5000]]))


from sklearn.externals import joblib

joblib.dump(reg,'model_joblib')

mj=joblib.load('model_joblib')

print(mj.predict([5000]))

