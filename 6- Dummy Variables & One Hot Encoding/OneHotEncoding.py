import pandas as pd

df=pd.read_csv("homeprices.csv")

dummies = pd.get_dummies(df.town)


merged= pd.concat([df,dummies],axis='columns')


final = merged.drop(['town','west windsor'],axis='columns')

#from sklearn import linear_model

#model = linear_model.LinearRegression()
#X= final.drop('price',axis='columns')
#Y= final.price

#model.fit(X,Y)

#print(model.predict([[3400,0,0]]))
#print(model.score(X,Y))

from sklearn.preprocessing import OneHotEncoder

X=df[['town','area']].values

ohe=OneHotEncoder(categorical_features=[0])

print(ohe.fit_transform(X).toarray())