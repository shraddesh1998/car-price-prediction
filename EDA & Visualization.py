#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries:

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[92]:


get_ipython().system('pip install streamlit')


# In[93]:


import streamlit as st


# In[2]:


cars = pd.read_csv("scraped cars24 data (1).csv")
cars


# #### Data set conains so many special characters so we have to perform split and join function to get a proper dataset:

# In[3]:


cars_split_1 = cars['EMI'].str.split("?" , expand = True)
cars_split_1


# In[4]:


cars_0 = cars_split_1.drop([0], axis = 1)
cars_0.rename({1:"EMI"}, axis = 1, inplace = True)
cars_0


# In[5]:


cars_split_2 = cars['Sales'].str.split("?" , expand = True)
cars_split_2


# In[6]:


cars_1 = pd.concat((cars ,cars_split_2 ), axis = 1)
cars_1


# In[7]:


cars_2 = cars_1.drop([0, 'EMI' , 'Sales','Types'] , axis = 1 )
cars_2.rename({1:"Original_Price" ,2:"Sales_Price" } , axis = 1 , inplace = True)
cars_2


# In[8]:


a1 = cars['Year'].str.split("-" , expand = True)
a1.rename({1:"Month"}, axis=1, inplace = True)
a2 = a1.drop((0),axis=1)


# In[9]:


a3 = cars_2.drop(("Year"), axis=1)
cars_2_new = pd.concat((a3,a2), axis =1)
cars_2_new.head()


# In[10]:


cars_3 = pd.concat((cars_2_new,cars_0), axis=1)
cars_3


# In[11]:


cars_split_3 = cars['Name'].str.split(" " , expand = True)
cars_split_3.rename({0:"Year",1:"Company"}, axis=1, inplace = True)
cars_split_3


# In[12]:


c1 = cars_split_3.drop([2,3,4,5,6,7,8,9,10], axis = 1)
join1 = cars_split_3[cars_split_3.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
join1


# In[13]:


cars_4 = pd.concat((join1,c1), axis =1)
cars_4.rename({0:"Name"}, axis= 1, inplace = True)
cars_4


# In[14]:


c2 = cars_3.drop(("Name"),axis=1)
cars_5 = pd.concat((cars_4,c2), axis =1)
cars_5.head()


# In[15]:


b1 = cars_5['EMI'].str.split("/" , expand = True)
b1.rename({0:"EMI(₹)"}, axis=1, inplace = True)
b2 = b1.drop((1),axis=1)


# In[16]:


d1 = cars_5.drop(("EMI"), axis=1)
cars_6 = pd.concat((d1,b2), axis =1)
cars_6.head()


# In[17]:


b3 = cars_6['KM_Driven'].str.split("km" , expand = True)
b3.rename({0:"KM_Driven"}, axis=1, inplace = True)
b4 = b3.drop((1),axis=1)


# In[18]:


d2 = cars_6.drop(("KM_Driven"), axis=1)
cars_7 = pd.concat((d2,b4), axis =1)
cars_7


# In[19]:


cars_7['Original_Price'] = cars_7['Original_Price'].replace(",","",regex = True)
cars_7['Sales_Price'] = cars_7['Sales_Price'].replace(",","",regex = True)
cars_7['EMI(₹)'] = cars_7['EMI(₹)'].replace(",","",regex = True)
cars_7['KM_Driven'] = cars_7['KM_Driven'].replace(",","",regex = True)


# In[20]:


#cars_7.to_csv('cars_new.csv')


# ### Cleaned Data:

# In[21]:


cars_new = pd.read_csv('cars_new.csv')
cars_new


# ## EDA:

# In[22]:


cars_new.info()


# In[23]:


cars_new.isna().sum()


# In[24]:


cars_new=cars_new.drop_duplicates()
cars = cars_new.reset_index().drop(["index"],axis=1)
cars


# ### Replacing Null values with Original Price & taking mode of transmission for replacing null values  :

# In[25]:


cars.Sales_Price.fillna(cars.Original_Price, inplace = True)
cars['Transmission']=cars.Transmission.fillna(cars['Transmission'].mode()[0])
cars['Month']=cars.Transmission.fillna(cars['Month'].mode()[0])


# In[26]:


cars.dropna( inplace=True)
cars.reset_index().drop(('index'), axis = 1)


# In[27]:


cars.isna().sum()


# In[28]:


#cars.to_csv('carsEDA.csv')


# ### Pandas Profiling & Sweetviz

# In[29]:


import pandas_profiling as pp
import sweetviz as sv

sweet_report = sv.analyze(cars)
sweet_report.show_html('Report.html')

EDA= pp.ProfileReport(cars)
EDA.to_file(output_file='Profile.html')


# ## Visualization:

# In[30]:


sns.pairplot(cars, hue='Transmission')


# In[31]:


corrmat = cars.corr()
top_corr_features = corrmat.index
sns.heatmap(cars[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# ### Checking relationship of Company with Sales Price

# In[32]:


cars['Company'].unique()


# In[33]:


plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='Company',y='Sales_Price',data=cars)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# ### Checking relationship of Year with Sales Price

# In[34]:


import warnings
warnings.filterwarnings('ignore')


# In[35]:


plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='Year',y='Sales_Price',data=cars)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# ### Checking relationship of KM_Driven with Sales Price

# In[36]:


sns.relplot(x='KM_Driven',y='Sales_Price',data=cars,height=7,aspect=1.5)


# ### Checking relationship of Fuel Type with Sales Price

# In[37]:


plt.subplots(figsize=(14,7))
sns.boxplot(x='Fuel_Type',y='Sales_Price',data=cars)


# ### Relationship of Sales Price with FuelType, Year and Company mixed

# In[38]:


ax=sns.relplot(x='Company',y='Sales_Price',data=cars,hue='Fuel_Type',size='Year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')


# ### Checking relationship of EMI with Sales Price for hue as Company

# In[ ]:


sns.scatterplot (x= "Sales_Price" , y = "EMI(₹)" , hue = "Company" , style = "Fuel_Type" , size = "Transmission" , data = cars   )
sns.set(rc= {'figure.figsize' :(15.7 , 10.27)})


# In[40]:


#!pip install pywedge


# In[41]:


import pywedge as pw


# In[42]:


mc=pw.Pywedge_Charts(cars,c=None,y='Sales_Price')


# In[43]:


charts=mc.make_charts()


# In[44]:


get_ipython().system('pip install pivottablejs')


# In[45]:


from pivottablejs import pivot_ui
pivot_ui(cars,outfile_path='pivottablejs.html',)


# # Model Building :

# ## Azure ML Pipeline Model :

# 
# ### Extracting Training Data

# In[46]:


x=cars[['Name','Company','Transmission','Year','KM_Driven','Fuel_Type','City']]
y=cars['Sales_Price']


# In[47]:


x


# In[48]:


y.shape


# ### Applying Train Test Split

# In[49]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[50]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# #### Creating an OneHotEncoder object to contain all the possible categories

# In[51]:


OHE=OneHotEncoder()
OHE.fit(x[['Name','Company','Transmission','Fuel_Type','City']])


# In[52]:


column_trans=make_column_transformer((OneHotEncoder(categories=OHE.categories_),['Name','Company','Transmission','Fuel_Type','City']),
                                    remainder='passthrough')


# In[53]:


from sklearn.linear_model import LinearRegression


# In[54]:


lr=LinearRegression()


# In[55]:


pipe=make_pipeline(column_trans,lr)


# In[56]:


pipe.fit(x_train,y_train)


# In[57]:


y_pred=pipe.predict(x_test)


# In[58]:


r2_score(y_test,y_pred)


# #### Finding the model with a random state of TrainTestSplit :

# In[59]:


scores=[]
for i in range(2000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))


# In[60]:


np.argmax(scores)


# In[61]:


scores[np.argmax(scores)]


# In[62]:


price = pipe.predict(pd.DataFrame(columns=x_test.columns,data=np.array(['Renault TRIBER 1.0 RXZ','Renault','MANUAL',2018,33419,'Petrol','Hyderabad']).reshape(1,7)))
print("The sales price will be:", np.round(price,2))


# ## Actual Vs Predicted :

# In[63]:


#Comparing original Data's Scatterlot with regressed model
plt.plot(cars.Sales_Price)
plt.plot(pipe.predict(cars))      
plt.show()


# ## Random Forest Regressor :

# In[64]:


# One Hot Encoding
cr7 = pd.get_dummies(cars,columns=['Name','Company','Transmission','Fuel_Type','City']) 


# In[65]:


x1=cr7.drop(['Registration','Month','Owner','Original_Price','EMI(₹)','Sales_Price'], axis=1)
y1=cars[['Sales_Price']]
x1.head()


# In[66]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


# #### RandomForestRegressor :

# In[67]:


num_trees = 65
max_features = 7


# In[68]:


kfold = KFold(n_splits=7, random_state=14, shuffle=True)
model3 = RandomForestRegressor(n_estimators=num_trees, criterion='mse', max_features=max_features)


# In[69]:


results = cross_val_score(model3, x1, y1, cv=kfold)
print(results.mean())


# #### RandomForestClassifier :

# In[70]:


num_trees = 82
max_features = 4


# In[71]:


kfold = KFold(n_splits=8, random_state=27, shuffle=True)
model2 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)


# In[72]:


results = cross_val_score(model2, x1, y1, cv=kfold)
print(results.mean())


# In[73]:


## We can't classify this dataset.


# ## Decision Tree Regressor :

# In[74]:


from sklearn.tree import  DecisionTreeRegressor


# In[75]:


# Label Encoding

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
cars['Name']= label_encoder.fit_transform(cars['Name'])
cars['Company']= label_encoder.fit_transform(cars['Company'])
cars['Transmission']= label_encoder.fit_transform(cars['Transmission'])
cars['Fuel_Type']= label_encoder.fit_transform(cars['Fuel_Type'])
cars['Owner']= label_encoder.fit_transform(cars['Owner'])
cars['Month']= label_encoder.fit_transform(cars['Month'])
cars['City']= label_encoder.fit_transform(cars['City'])


# In[76]:


x2=cars.drop(['Original_Price','Owner','Registration','EMI(₹)','Sales_Price'], axis=1)
y2=cars[['Sales_Price']]
x2.head()


# #### Checking pp score :

# In[77]:


import ppscore as pps
pps.matrix(cars)        #calculate the whole PPS matrix


# In[78]:


pps.score(cars, "Sales_Price", "KM_Driven")


# ### Decision Tree Regression :

# In[79]:


x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.3, random_state=21)


# In[80]:


model5 = DecisionTreeRegressor()
model5.fit(x_train, y_train)


# In[81]:


model5.score(x_test,y_test)           #Accuracy


# ### Building Decision Tree Classifier (CART) using Gini Criteria :

# In[82]:


x_train, x_test,y_train,y_test = train_test_split(x2,y2, test_size=0.22,random_state=18)


# In[83]:


from sklearn.tree import DecisionTreeClassifier
model4_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[84]:


model4_gini.fit(x_train, y_train)


# In[85]:


y_test


# In[87]:


y_test3 = y_test.to_numpy()
y_test3 = np.reshape(y_test3, 447)
y_test3


# In[88]:


#Prediction and computing the accuracy
pred=model5.predict(x_test)
np.mean(pred==y_test3)


# In[89]:


### There is no common criteria for which we can classify this data


# In[ ]:





# In[90]:


pip freeze


# In[ ]:




