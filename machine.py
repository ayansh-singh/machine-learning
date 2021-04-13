import pandas as pd
import quandl
import math,numpy as np, datetime
import sklearn
from sklearn import linear_model
import pickle
quandl.ApiConfig.api_key = 'ayphy5_xMYJusihkTtNK'
df = quandl.get(("WSE/STOMIL"),authtoken='ayphy5_xMYJusihkTtNK',collapse="daily")
df=df[['Open','High','Low','Close','Volume']]
df['hl_pct']=((df['High']-df['Low'])/df['High'])*100
df['daily_pct_change']=((df['Close']-df['Open'])/df['Open'])*100
forecast_col='Close'
forecast_out=int(math.ceil(0.01*len(df)))
df['label']= df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
x = np.array(df.drop([forecast_col],1))
x = sklearn.preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df[forecast_col])
y = y[:-forecast_out]
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)
with open('LinearRegression','wb') as f:
    pickle.dump(linear,f)
saveddf=open('LinearRegression','rb')
linear=pickle.load(saveddf)
acc=linear.score(x_test,y_test)
forecast_set=linear.predict(x_lately)
print(forecast_set,acc)
