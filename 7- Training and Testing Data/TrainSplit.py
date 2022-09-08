import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('carprices.csv')

plt.scatter(df['Mileage'],df['Sell Price($)'])
#plt.show()
plt.scatter(df['Age(yrs)'],df['Sell Price($)'])
#plt.show()

X=df[['Mileage','Age(yrs)']]
y=df[['Sell Price($)']]

from sklearn.model_selection import train_test_split

xTrain,xTest,yTrain,yTest=train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

clf=LinearRegression()
clf.fit(xTrain,yTrain)

print(clf.predict(xTest))
print(yTest)
print(clf.score(xTest,yTest))


