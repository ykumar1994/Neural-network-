#!/usr/bin/env python
# coding: utf-8

# # AML 3104 - Neural Networks and Deep Learning 02 (DSMM Group 2)
# ## Final Neural network Project ( Term 3)    
# 
# ### Category: Recurrent Neural Network and LSTMs
# ### Topic: Time-series Forecasting
# 
# ![image-2.png](attachment:image-2.png)
# 
# ### Group members:
# #### 1. Yogesh Kumar( C0852435)
# #### 2. Prashant Kumar Vashney (C0850851)
# #### 3. Bishal Subedi (C0852384)
# #### 4. Husan Preet Singh Gill (C0851931)
# 

# # Stock Market Analysis and  Prediction using LSTM
# 

# ## Importing libraries 

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[2]:


get_ipython().system('pip install yfinance')
import yfinance as yf

get_ipython().system('pip install pandas-datareader')

import pandas_datareader
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr


# ## Data extraction and preparation

# In[3]:


yf.pdr_override()

end_date = datetime.now()
start_date = datetime(end_date.year - 1, end_date.month, end_date.day)

APPL = pdr.DataReader('AAPL', start=start_date, end=end_date)
GOOG= pdr.DataReader('GOOG', start=start_date, end=end_date)
MSFT = pdr.DataReader('MSFT', start=start_date, end=end_date)
AMZN = pdr.DataReader('AMZN', start=start_date, end=end_date)

print(APPL)


# In[4]:


APPL["stock_name"]="Apple"
GOOG["stock_name"]="Google"
MSFT["stock_name"]="Microsoft"
AMZN["stock_name"]="Amazon"


# In[5]:


stock_data=[APPL,GOOG,MSFT,AMZN]
df=pd.concat([APPL,GOOG,MSFT,AMZN],axis=0)
df


# ## Preliminary Data Analysis

# ### Descriptive Statistics

# In[6]:


stock_name_list=["Apple","Google","Microsoft","Amazon"]
for stk_name in stock_name_list:
    print("stats for stock {} are as follows".format(stk_name),"\n",
    df.where(df['stock_name']==stk_name).describe(),"\n")


# ### Information About the Data

# In[7]:


for stk_name in stock_name_list:
    print("stats for stock {} are as follows".format(stk_name),"\n",
    df.where(df['stock_name']==stk_name).info(),"\n")


# ### Closing Price
# #### The closing price is the last price at which the stock is traded during the regular trading day. A stock’s closing price is the standard benchmark used by investors to track its performance over time.

# In[8]:


# Let's see a historical view of the closing price
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(stock_data, 1):
    plt.subplot(2, 2, i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {stock_name_list[i - 1]}")
    
plt.tight_layout()


# In[9]:


for company in stock_data:
    company['Daily Return'] = company['Adj Close'].pct_change()


# In[10]:


# lets check the daily return for each company
plt.figure(figsize=(12, 9))

for i, company in enumerate(stock_data, 1):
    plt.subplot(2, 2, i)
    company['Daily Return'].hist(bins=50)
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(f'{stock_name_list[i - 1]}')
    
plt.tight_layout()


# ### Correlation between different stocks closing prices

# In[11]:


sr=pd.concat([APPL['Adj Close'],GOOG['Adj Close'],MSFT['Adj Close'],AMZN['Adj Close']],axis=1)
sr.columns=stock_name_list
sr.dropna()


# In[12]:


cp=pd.concat([APPL['Daily Return'],GOOG['Daily Return'],MSFT['Daily Return'],AMZN['Daily Return']],axis=1)
cp.columns=stock_name_list
cp.dropna()


# In[13]:


plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.heatmap(sr.corr(), annot=True, cmap='summer')
plt.title('Correlation of stock return')

plt.subplot(2, 2, 2)
sns.heatmap(cp.corr(), annot=True, cmap='summer')
plt.title('Correlation of stock closing price')


# ##### From above heatmap we can say that Amazon and Google stocks are highly correlated . Also Amazon and Microsoft stocks are least correlated

# ### Predicting the closing price stock price of APPLE inc:

# In[14]:


# Getting the stock quote for last 10 years 
df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())

df


# In[15]:


plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price ($)', fontsize=18)
plt.show()


# In[16]:


# Create a new dataframe with only the 'Close column 
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

training_data_len


# In[17]:


# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[18]:


# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape


# In[19]:




# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])


# In[20]:


# Defining callbacks

# Save weights only for best model
checkpointer = ModelCheckpoint(filepath = 'weights_best.hdf5', 
                               verbose = 2, 
                               save_best_only = True)
#training the model
history=model.fit(x_train, 
          y_train, 
          epochs=25, 
          batch_size = 32,
          callbacks = [checkpointer])


# In[21]:


# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse


# In[22]:


# Plotting the taining and predicted data 
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[23]:


print(f"the LSTM model have accuracy {history.history['accuracy'][0]*100000}% precdicting close prices values of Apple stock")


# In[24]:


loss=history.history['loss']
acc=history.history['accuracy']

import matplotlib.pyplot as plt

plt.plot(loss, label='loss')
plt.plot(acc, label='accuracy')
plt.ylabel('accuracy , Loss')
plt.xlabel('Epoch')
plt.title('Accuracy and loss plot')
plt.legend()

plt.show()


# In[25]:


#below are the predictions values
valid


# ### Above are the actual close prices values for APPLE stock and values predicted by LSTM model. we can say that model is predicting well with 75.9% accuracy.

# In[ ]:




