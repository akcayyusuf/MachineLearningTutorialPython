import numpy as np

def gD(x,y):
    mCurr=bCurr=0
    iterations=100000
    n=len(x)
    learningRate=0.001
    for i in range(iterations):
        y_pred=mCurr*x+bCurr
        cost=(1/n)*sum([val**2 for val in (y-y_pred) ])
        md=-(2/n)*sum(x*(y-y_pred))
        bd=-(2/n)*sum(y-y_pred)

        mCurr=mCurr-learningRate*md
        bCurr=bCurr-learningRate*bd
        print("m {}, b {}, Cost {} iteration {}".format(mCurr,bCurr,cost,i))
    pass

x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gD(x,y)