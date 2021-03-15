#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.stats as sc


# In[2]:


from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# In[3]:


import pip
get_ipython().run_line_magic('pip', 'install "yfinance"')
get_ipython().run_line_magic('pip', 'install "yahoofinancials"')


# In[4]:


get_ipython().run_line_magic('pip', 'install pandas_datareader')


# In[5]:


from pandas_datareader import data as pdr


# ### Initial research and analysis

# In[6]:


import yfinance as yf
from yahoofinancials import YahooFinancials
from datetime import date
today = date.today()


# In[7]:


def getd (symbol):
    tickerSymbol = symbol
    tickerData = yf.Ticker(tickerSymbol)
    start_date='2015-01-01'
    d = pdr.get_data_yahoo(symbol, start=start_date, end=today)
    return d


# In[8]:


#check our data 
df=getd ("^VIX")
df


# In[10]:


# compute volatility using Pandas rolling and std methods, the trading days is set to 2 weeks 
TRADING_DAYS = 30*6
symbol2="^GSPC"
df2=getd (symbol2)
returns2 = np.log(df2['Close']/df2['Close'].shift(1))
returns2.fillna(0, inplace=True)
volatility2 = returns2.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)

TRADING_DAYS = 30*6
symbol="^VIX"
df=getd (symbol)
vix=df['Close']


# In[11]:


fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(1, 1, 1)
vix.plot(ax=ax1,label=symbol)
#volatility2.plot(ax=ax1, label=symbol2)
ax1.set_xlabel('Date')
ax1.set_ylabel('Volatility')
plt.legend(loc="upper left")
#plt.ylim(-1.5, 2.0)
ax1.set_title('Price of {0} for {2} days'.format(symbol,symbol2,TRADING_DAYS ))
plt.show()


# In[12]:


fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(1, 1, 1)
#vix.plot(ax=ax1,label=symbol)
volatility2.plot(ax=ax1, label=symbol2)
ax1.set_xlabel('Date')
ax1.set_ylabel('Volatility')
plt.legend(loc="upper left")
#plt.ylim(-1.5, 2.0)
ax1.set_title('Volatility of {0} for {1} days'.format(symbol2,TRADING_DAYS))
plt.show()


# In[13]:


# compute sharpe ratio using Pandas rolling and std methods
risk_free_rate=0

TRADING_DAYS = 30*6
symbol="^GSPC"
df=getd (symbol)
returns = np.log(df['Close']/df['Close'].shift(1))
returns.fillna(0, inplace=True)
volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)
sharpe_ratio = (returns.mean() - risk_free_rate)/volatility
sharpe_ratio.tail()

fig = plt.figure(figsize=(15, 7))
ax3 = fig.add_subplot(1, 1, 1)
sharpe_ratio.plot(ax=ax3)
ax3.set_xlabel('Date')
ax3.set_ylabel('Sharpe ratio')
ax3.set_title('Sharpe ratio with the {0} days volatility for {1}'.format(TRADING_DAYS,symbol))
plt.show()


# In[14]:


class s:
    def getd (symbol):
        tickerSymbol = symbol
        tickerData = yf.Ticker(tickerSymbol)
        d=tickerData.history(period='1d', start='2010-1-1', end=today)
        return d


# In[15]:


class pair:
    #30 day moving average plot based on 95 and 5 percentiles 
    def ma(symbol1,symbol2):
        def zscore(numbers):
            return (numbers - numbers.mean())/np.std(numbers)
        d1=s.getd(symbol1) 
        d2=s.getd(symbol2)
        spread= d1['Close']-d2['Close']
        ma1 = spread.rolling(1).mean()
        ma30 = spread.rolling(30).mean()
        std30 = spread.rolling(30).std()
        zscore30=(ma1 - ma30)/std30
        z30=zscore30.dropna()
        n1= np.percentile(z30,95)
        n2= np.percentile(z30,5)
        zscore30.plot()
        plt.axhline(zscore30.mean())
        plt.axhline(n1,color='r',ls='--')
        plt.axhline(n2, color='r', ls='--')


# In[16]:


pair.ma("^GSPC", "^VIX") 


# In[17]:


win=50
win2=200
short_rolling = volatility2.rolling(window=win).mean()
long_rolling = volatility2.rolling(window=win2).mean()
#short_rolling.head(20)


# In[18]:


fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(1, 1, 1)
short_rolling.plot(ax=ax1,label="S&P 500 short rolling")
long_rolling.plot(ax=ax1,label="S&P 500 long rolling")
ax1.set_xlabel('Date')
ax1.set_ylabel('Volatility')
plt.legend(loc="upper left")
ax1.set_title('{0} and {1} days volatility moving average of {2}'.format(win,win2,symbol2))
plt.show()


# In[19]:


win=50
win2=200
short_rolling2 = vix.rolling(window=win).mean()
long_rolling2 = vix.rolling(window=win2).mean()


# In[20]:


fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(1, 1, 1)
short_rolling2.plot(ax=ax1,label="VIX short rolling")
long_rolling2.plot(ax=ax1,label="VIX long rolling")
ax1.set_xlabel('Date')
ax1.set_ylabel('Volatility')
plt.legend(loc="upper left")
ax1.set_title('{0} and {1} days moving average of {2}'.format(win,win2,"VIX"))
plt.show()


# ### Trading strategy

# In[21]:


get_ipython().run_line_magic('pip', 'install quandl')


# In[22]:


symbol="^VIX"
df=getd (symbol)
vix=df['Close']
win=10
win2=30
signals=pd.DataFrame(index=vix.index)
signals['signal']=0.0
signals['short_mavg']= vix.rolling(window=win,min_periods=1,center=False).mean()
signals['long_mavg']= vix.rolling(window=win2,min_periods=1,center=False).mean()
signals['signal'][win:]=np.where(signals['short_mavg'][win:]
                                         > signals['long_mavg'][win:], 1.0, 0.0 )

signals['positions']=signals['signal'].diff()
print(signals)


# In[23]:


fig = plt.figure(figsize=(20, 15))
ax1 = fig.add_subplot(111,ylabel='Price in $')
vix.plot(ax=ax1,color='black',lw=2.)
signals[['short_mavg','long_mavg']].plot(ax=ax1,lw=2.0)
#plot the buy signals 
ax1.plot(signals.loc[signals.positions==1.0].index,
        signals.short_mavg[signals.positions == 1.0],
        '^',markersize=10,color='g')

#plot the sell signals
ax1.plot(signals.loc[signals.positions==-1.0].index,
        signals.short_mavg[signals.positions == -1.0],
        'v',markersize=10,color='r')

plt.show()


# ### backtesting

# In[24]:


initial_cap=float(100_000)
positions=pd.DataFrame(index=signals.index).fillna(0.0)

#but 1000 shares
positions['Position in VIX']=1_000*signals['positions']

#initialize the portfolio with shares owned
portfolio=positions.multiply(vix,axis=0)

#store difference in shares owned 
pos_diff=positions.diff()

#add 'holding'
portfolio['holdings']=(positions.multiply(vix,axis=0)).sum(axis=1)

#add 'cash'
portfolio['cash']=initial_cap - (pos_diff.multiply(vix,axis=0)).sum(axis=1).cumsum()

#add 'total'
portfolio['total']=portfolio['cash'] + portfolio['holdings']

#add 'return'
portfolio['return']=portfolio['total'].pct_change()

print(portfolio.tail())


# In[25]:


positions['Position in VIX']


# In[26]:


fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(111,ylabel='Return')
portfolio['total'].plot(ax=ax1,lw=1.)
ax1.set_title('Total Asset of Portfolio')
plt.show()


# In[27]:


fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(111,ylabel='Return')
portfolio['return'].plot(ax=ax1,lw=1.)
plt.show()


# In[34]:


from statistics import mean
mean(portfolio['return'])


# In[35]:


from numpy import mean
# compute sharpe ratio using Pandas rolling and std methods
risk_free_rate=0

TRADING_DAYS = 252
returns = portfolio['return']
returns.fillna(0, inplace=True)
volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)
sharpe_ratio = (returns.mean() - risk_free_rate)/volatility
avg_sharpe=mean(sharpe_ratio)
avg_sharpe


# In[36]:


fig = plt.figure(figsize=(15, 7))
ax3 = fig.add_subplot(1, 1, 1)
sharpe_ratio.plot(ax=ax3)
ax3.set_xlabel('Date')
ax3.set_ylabel('Sharpe ratio')
ax3.set_title('Sharpe ratio with the {0} days volatility for {1}'.format(TRADING_DAYS,symbol))


# In[37]:


print("Portfolio total value as of {0}: ${1}".format(today,(round((portfolio['total'][-1]),2))))
abs_return=((portfolio['total'][-1]/float(100_000))-float(1))*100
print("Absolute return as of {0}: {1}%".format(today,round(abs_return,2)))


# In[43]:


get_ipython().system('export PATH=/Library/TeX/texbin:$PATH')


# In[42]:


jupyter nbconvert your_notebook.ipynb --to pdf


# In[ ]:




