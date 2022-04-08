#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf

# For time stamps
from datetime import datetime


# In[29]:


tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# Set up End and Start times for data grab
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']


end = datetime.now()
start = datetime(end.year - 4, end.month, end.day)


globals()['GOOG'] = yf.download('GOOG', start, end)


# In[30]:


company_list = [ GOOG]
company_name = [ "GOOGLE"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
df = pd.concat(company_list, axis=0)
df.tail(10)


# In[31]:


df1=df.reset_index()['Close']


# In[32]:


df1


# In[33]:


#plt.plot(df1)


# In[34]:


import matplotlib.pyplot as plt
plt.subplots_adjust(top=1.25, bottom=1.2)
company['Adj Close'].plot()
plt.ylabel('Adj Close')
plt.xlabel(None)
plt.title(f"Closing Price of GOOGLE")


# In[35]:


df1


# In[36]:


# make changes to make plot before this
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[37]:


print(df1)


# In[38]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[39]:


training_size,test_size


# In[40]:


train_data


# In[41]:



# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


# In[42]:



time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[74]:


print(X_train.shape), print(y_train.shape)


# In[48]:


np.size(y_train)


# In[75]:


print(X_test.shape), print(ytest.shape)


# In[50]:


np.size(ytest)


# In[47]:


print(X_train)


# In[76]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[77]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[78]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')


# In[79]:


model.summary()


# In[80]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[165]:


import tensorflow as tf
tf.__version__


# In[166]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[167]:


math.sqrt(mean_squared_error(ytest,test_predict))


# In[168]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)



plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend(['Actual value ', 'Train prediction', 'Test Prediction'], loc='lower right')

plt.show()


# In[186]:


print(test_data)


# In[190]:


x_input=test_data[254:].reshape(1,-1)
x_input.shape


# In[191]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input


# In[192]:


# demonstrate prediction for next 30 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[193]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)
import matplotlib.pyplot as plt
len(df1)


# In[209]:


plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel(' from 100 + 30 days prediction', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)





plt.legend(['Actual value ', 'Test Prediction'], loc='lower right')

plt.plot(day_new,scaler.inverse_transform(df1[909:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.show()


# In[211]:


df3=df1.tolist()
df3.extend(lst_output)
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel(' when graph is adjusted', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

plt.plot(df3[900:])
plt.show()


# In[214]:


df3=scaler.inverse_transform(df3).tolist()
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel(' Graph from 3 years + 30 days predicted', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(df3)


# In[ ]:




