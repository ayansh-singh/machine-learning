import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
df=pd.read_csv('breast-cancer-wisconsin.data')
df.drop(['id'],1,inplace=True)
df.replace('?',-99999,inplace=True)
x = df.drop(['type'],1)
y=df['type']
x_train, x_test, y_train, y_test=model_selection.train_test_split(x,y,test_size=0.2)
clf= neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
