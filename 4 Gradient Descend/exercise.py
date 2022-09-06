import math

import numpy as np
import pandas as pd
from sklearn import linear_model

def gD(x,y):
    costOLD=1000
    mCurr=bCurr=0
    iterations=1000000
    n=len(x)
    learningRate=0.0001
    for i in range(iterations):
        y_pred=mCurr*x+bCurr
        cost=(1/n)*sum([val**2 for val in (y-y_pred) ])
        md=-(2/n)*sum(x*(y-y_pred))
        bd=-(2/n)*sum(y-y_pred)

        mCurr=mCurr-learningRate*md
        bCurr=bCurr-learningRate*bd
        print("m {}, b {}, Cost {} iteration {}".format(mCurr,bCurr,cost,i))

        if math.isclose(cost,costOLD,rel_tol=1e-20):
            print("buldum")
            break
        costOLD=cost

    return mCurr,bCurr

def sklearnOutput():
    df = pd.read_csv("test_scores.csv")
    reg = linear_model.LinearRegression()
    reg.fit(df[['math']],df.cs)
    return reg.coef_, reg.intercept_





df = pd.read_csv('test_scores.csv')
x=np.array(df.math)
y=np.array(df.cs)


print(gD(x,y))
print(sklearnOutput())
