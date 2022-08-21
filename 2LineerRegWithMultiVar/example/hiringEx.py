import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from word2number import w2n

df=pd.read_csv("hiring.csv")

df.experience=df.experience.fillna('zero')
df.experience = df.experience.apply(w2n.word_to_num)
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean())

reg=linear_model.LinearRegression().fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])
print(reg.predict([[20,10,10]]))