#!/usr/bin/env python
# coding: utf-8

# # Stock Market Analysis 📈 and  Prediction using LSTM

# In[16]:


import pandas as pd
import numpy as np
from datetime import datetime


# In[10]:


get_ipython().system('pip install yfinance')
import yfinance as yf

get_ipython().system('pip install pandas-datareader')

import pandas_datareader
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr


# In[26]:


yf.pdr_override()

end_date = datetime.now()
start_date = datetime(end.year - 1, end.month, end.day)

AAPL = pdr.DataReader('AAPL', start=start_date, end=end_date)
GOOG= pdr.DataReader('GOOG', start=start_date, end=end_date)
MSFT = pdr.DataReader('MSFT', start=start_date, end=end_date)
AMZN = pdr.DataReader('AMZN', start=start_date, end=end_date)

print(AAPL)


# In[29]:


APPL["stock_name"]="Apple"
GOOG["stock_name"]="Google"
MSFT ["stock_name"]="Microsoft"
AMZN["stock_name"]="Amazon"


# In[32]:


df=pd.concat([APPL,GOOG,MSFT,AMZN],axis=0)
print(df)


# In[ ]:




