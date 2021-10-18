#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[90]:


ds=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/Advertising.csv")


# In[91]:


ds


# In[92]:


df=pd.DataFrame(ds)
df


# In[93]:


df.drop(df.iloc[:, 0:-4],
                      axis = 1,inplace=True)
df


# In[94]:


df.shape


# In[95]:


df.dtypes


# In[96]:


df.columns


# In[97]:


df.head()


# In[98]:


df.tail()


# In[99]:


df["TV"].value_counts()


# In[100]:


df.info()


# In[101]:


df["newspaper"].value_counts()


# In[102]:


df["radio"].value_counts()


# In[103]:


df.columns.value_counts()


# In[104]:


#checking out the null values



df.isnull()


# In[105]:


df.isnull().sum()


# In[106]:


sns.heatmap(df.isnull())


# In[107]:


#no null values here so no need to remove or replace any nan value.


# In[108]:


#highlights on statistics


# In[109]:


df.describe()


# In[110]:


#conclusions=
#1)THE DATA BELONGIG TV COLUMN IS HGHLY DISTRIBUTED AS THE STANDARD DEVIATION IS MORE COMPARED TO THE OTHER COLUMNS.
#2)NEWSPAPER COLUMN HAS HIGHTEST DIFF BETWEEN 75% AND MAX VALUE SO POSSIBILITY OF OUTLIERS IN THE SAME.
#3)NWESPAPER DATA HAS MORE MEAN SO PRESENCE OF SKEWNESS IS THERE
#4)LOT AMOUNT IS SPENT FOR TV AD COMPARED TO OTHER TWO AND LOWEST ON RADIO.
#5)THERE IS AN AVERAGE SALES OF FOURTEEN DOLLARS. 


# In[111]:


#Lets do some univariate analysis


# In[112]:


sns.distplot(df["TV"])


# In[113]:


sns.boxplot(df["newspaper"])


# In[114]:


sns.boxplot(df["radio"])


# In[115]:


sns.boxplot(df["TV"])


# In[116]:


#no more outliers are present in the dataset.only present in newspaper so even if not considered thats ok.


# EDA PROCESS

# In[117]:


corelation=df.corr()
corelation


# In[118]:


sns.heatmap(corelation,annot=True)


# In[119]:


sns.pairplot(df)
plt.show()


# In[120]:


#TV is highly correlated with sales and radio is least.


# In[121]:


sns.distplot(df["newspaper"])


# In[122]:


sns.histplot(ds["sales"],bins=20)


# In[ ]:


sns.regplot(x="sales",y="TV",data=ds)


# In[ ]:


#getting x.y values to split into input and target variables


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x=df.iloc[:,:-1]
x


# In[ ]:


y=df.iloc[:,-1]
y


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[ ]:


LR=LinearRegression(fit_intercept=True)
LR.fit(xtrain,ytrain)
ypred=LR.predict(xtest)


# In[ ]:


LR.intercept_


# In[ ]:


LR.coef_


# In[ ]:


ROOT_MEAN_SQUARED_ERROR=np.sqrt(metrics.mean_squared_error(ytest,ypred))
ROOT_MEAN_SQUARED_ERROR


# In[ ]:


MEAN_ABSOLUTE_ERROR=np.sqrt(metrics.mean_squared_error(ytest,ypred))
MEAN_ABSOLUTE_ERROR


# In[ ]:


r2_score(ypred,ytest)


# regularisation

# In[ ]:


from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score
RD=Ridge(alpha=0.001,random_state=46)
RD.fit(xtrain,ytrain)
RD.score(xtrain,ytrain)


# In[ ]:


lss=Lasso(alpha=0.001,random_state=47)
lss.fit(xtrain,ytrain)
lss.score(xtrain,ytrain)


# In[ ]:


#both the models are giving same result


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


DTC=DecisionTreeRegressor(criterion="mse")
DTC.fit(xtrain,ytrain)
print("DTC",DTC.score(xtrain,ytrain))



DTCpredict=DTC.predict(xtest)
print("DTC r2_score",r2_score(ytest,DTCpredict))
print("MSE of DTC::",mean_squared_error(ytest,DTCpredict))
print("RMSE of DTC::",np.sqrt(mean_squared_error(ytest,DTCpredict)))


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


kn=KNeighborsRegressor()
kn.fit(xtrain,ytrain)
print("knn",kn.score(xtrain,ytrain))



knpredict=kn.predict(xtest)
print("kn r2_score",r2_score(ytest,knpredict))
print("MSE of kn::",mean_squared_error(ytest,knpredict))
print("RMSE of kn::",np.sqrt(mean_squared_error(ytest,knpredict)))


# In[ ]:


from sklearn.svm  import SVR


# In[ ]:


svr=SVR()
svr.fit(xtrain,ytrain)
print("svr",svr.score(xtrain,ytrain))



svrpredict=svr.predict(xtest)
print("svr r2_score",r2_score(ytest,svrpredict))
print("MSE of svr::",mean_squared_error(ytest,svrpredict))
print("RMSE of svr::",np.sqrt(mean_squared_error(ytest,svrpredict)))


# In[ ]:


#DTC model is performing better till now


# In[ ]:


from sklearn.linear_model  import SGDRegressor

sgd=SGDRegressor()
sgd.fit(xtrain,ytrain)
print("DTC score",sgd.score(xtrain,ytrain))
sgdpredict=sgd.predict(xtest)
print("DTCr2_score",r2_score(ytest,sgdpredict))
print("MSE of DTC::",mean_squared_error(ytest,sgdpredict))
print("RMSE of DTC::",np.sqrt(mean_squared_error(ytest,sgdpredict)))


# In[ ]:



from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=50,random_state=43)
rf.fit(xtrain,ytrain)
predrf=rf.predict(xtest)






print("rf score",rf.score(xtrain,ytrain))

print("rf r2_score",r2_score(ytest,predrf))


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
ADA=AdaBoostRegressor(n_estimators=50,random_state=43)
ADA.fit(xtrain,ytrain)
predADA=ADA.predict(xtest)



print("ADA score",ADA.score(xtrain,ytrain))

print("ADA r2_score",r2_score(ytest,predADA))


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
GB=GradientBoostingRegressor()
GB.fit(xtrain,ytrain)
predGB=GB.predict(xtest)




print("GB score",GB.score(xtrain,ytrain))

print("GB r2_score",r2_score(ytest,predGB))


# In[ ]:


from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute


# In[ ]:


cv = KFold(n_splits=10, random_state=1, shuffle=True)

#build multiple linear regression model
model = LinearRegression()

#use k-fold CV to evaluate model
scores = cross_val_score(model, x, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

#view mean absolute error
mean(absolute(scores))


# In[ ]:




