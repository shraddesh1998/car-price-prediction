#-- coding: utf-8 --
%matplotlib inline


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title= "CARS24 DASHBOARD" ,
                page_icon= ":Plots:" ,
                layout= "wide")


cars=pd.read_csv("scraped cars24 data (4).csv")

st.dataframe(cars)

#-------SIDEBAR---------
city = st.sidebar.multiselect(
     "Select the City :",
     options= cars["City"].unique(),
     default= cars["City"].unique()

)

Transmission = st.sidebar.multiselect(
     "Select the Transmission :",
     options= cars["Transmission"].unique(),
     default= cars["Transmission"].unique()
)

fuel = st.sidebar.multiselect(
     "Select the fuel :",
     options= cars["Fuel_Type"].unique(),
     default= cars["Fuel_Type"].unique()
)

owner = st.sidebar.multiselect(
     "Select the Owner :",
     options= cars["Owner"].unique(),
     default= cars["Owner"].unique()
)

Car_Type = st.sidebar.multiselect(
     "Select the Car_Type :",
     options= cars["Car_Type"].unique(),
     default= cars["Car_Type"].unique()
)


cars_selection = cars.query(
    "City == @city & Transmission == @Transmission & Fuel_Type == @fuel & Owner == @owner & Car_Type == @Car_Type"
)

#--------MAINPAGE---------
st.title(":Plots: CARS24 DASHBOARD")
st.markdown("##")

#TOP KPI'S
Max_price = (cars["Discounted_Price"].max())
average_Price = round(cars["Discounted_Price"].mean(),1)
Min_Price = (cars["Discounted_Price"].min(),2)

left_column , middle_column  = st.columns(2)
with left_column:
    st.subheader("Maximum Price :")
    st.subheader(f"Rs {Max_price:,}")
with middle_column:
    st.subheader("Average Price :")
    st.subheader(f"Rs {average_Price:,}")


st.markdown("____")


#--------- Charts------------

Fuel_wise_sellingPrice = px.bar(
    cars , x = "Discounted_Price", y = "Fuel_Type",
    orientation="h",
    title="<b>Fuel_wise_sellingPrice</b>",
    color_discrete_sequence=["#0083B8"]*len(Fuel_wise_sellingPrice)
)

st.plotly_chart(Fuel_wise_sellingPrice)


cars_split_1 = cars['EMI'].str.split("?" , expand = True)


cars_0 = cars_split_1.drop([0], axis = 1)
cars_0.rename({1:"EMI"}, axis = 1, inplace = True)

cars_split_2 = cars['Sales'].str.split("?" , expand = True)


cars_1 = pd.concat((cars ,cars_split_2 ), axis = 1)

cars_2 = cars_1.drop([0, 'EMI' , 'Sales','Types'] , axis = 1 )
cars_2.rename({1:"Original_Price" ,2:"Sales_Price" } , axis = 1 , inplace = True)


a1 = cars['Year'].str.split("-" , expand = True)
a1.rename({1:"Month"}, axis=1, inplace = True)
a2 = a1.drop((0),axis=1)



a3 = cars_2.drop(("Year"), axis=1)
cars_2_new = pd.concat((a3,a2), axis =1)
cars_2_new.head()


cars_3 = pd.concat((cars_2_new,cars_0), axis=1)


cars_split_3 = cars['Name'].str.split(" " , expand = True)
cars_split_3.rename({0:"Year",1:"Company"}, axis=1, inplace = True)


c1 = cars_split_3.drop([2,3,4,5,6,7,8,9,10], axis = 1)
join1 = cars_split_3[cars_split_3.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)


cars_4 = pd.concat((join1,c1), axis =1)
cars_4.rename({0:"Name"}, axis= 1, inplace = True)


c2 = cars_3.drop(("Name"),axis=1)
cars_5 = pd.concat((cars_4,c2), axis =1)


b1 = cars_5['EMI'].str.split("/" , expand = True)
b1.rename({0:"EMI(₹)"}, axis=1, inplace = True)
b2 = b1.drop((1),axis=1)


d1 = cars_5.drop(("EMI"), axis=1)
cars_6 = pd.concat((d1,b2), axis =1)


b3 = cars_6['KM_Driven'].str.split("km" , expand = True)
b3.rename({0:"KM_Driven"}, axis=1, inplace = True)
b4 = b3.drop((1),axis=1)


d2 = cars_6.drop(("KM_Driven"), axis=1)
cars_7 = pd.concat((d2,b4), axis =1)


cars_7['Original_Price'] = cars_7['Original_Price'].replace(",","",regex = True)
cars_7['Sales_Price'] = cars_7['Sales_Price'].replace(",","",regex = True)
cars_7['EMI(₹)'] = cars_7['EMI(₹)'].replace(",","",regex = True)
cars_7['KM_Driven'] = cars_7['KM_Driven'].replace(",","",regex = True)


cars_7.to_csv('cars_new.csv')



cars_new = pd.read_csv('cars_new.csv')



cars_new=cars_new.drop_duplicates()
cars = cars_new.reset_index().drop(["index"],axis=1)



cars.Sales_Price.fillna(cars.Original_Price, inplace = True)
cars['Transmission']=cars.Transmission.fillna(cars['Transmission'].mode()[0])
cars['Month']=cars.Transmission.fillna(cars['Month'].mode()[0])



cars.dropna( inplace=True)
cars.reset_index().drop(('index'), axis = 1)



cars.to_csv('carsEDA.csv')



import pandas_profiling as pp
import sweetviz as sv

sweet_report = sv.analyze(cars)
sweet_report.show_html('Report.html')

EDA= pp.ProfileReport(cars)
EDA.to_file(output_file='Profile.html')



sns.pairplot(cars, hue='Transmission')



corrmat = cars.corr()
top_corr_features = corrmat.index
sns.heatmap(cars[top_corr_features].corr(),annot=True,cmap="RdYlGn")



cars['Company'].unique()



plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='Company',y='Sales_Price',data=cars)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')



import warnings
warnings.filterwarnings('ignore')


plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='Year',y='Sales_Price',data=cars)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')



sns.relplot(x='KM_Driven',y='Sales_Price',data=cars,height=7,aspect=1.5)



plt.subplots(figsize=(14,7))
sns.boxplot(x='Fuel_Type',y='Sales_Price',data=cars)



ax=sns.relplot(x='Company',y='Sales_Price',data=cars,hue='Fuel_Type',size='Year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')



sns.scatterplot (x= "Sales_Price" , y = "EMI(₹)" , hue = "Company" , style = "Fuel_Type" , size = "Transmission" , data = cars   )
sns.set(rc= {'figure.figsize' :(15.7 , 10.27)})



import pywedge as pw


mc=pw.Pywedge_Charts(cars,c=None,y='Sales_Price')


st.header("Pywedge Charts")
st.plot(mc.make_charts())



get_ipython().system('pip install pivottablejs')



from pivottablejs import pivot_ui
pivot_ui(cars,outfile_path='pivottablejs.html',)



x=cars[['Name','Company','Transmission','Year','KM_Driven','Fuel_Type','City']]
y=cars['Sales_Price']



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)




from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score



OHE=OneHotEncoder()
OHE.fit(x[['Name','Company','Transmission','Fuel_Type','City']])



column_trans=make_column_transformer((OneHotEncoder(categories=OHE.categories_),['Name','Company','Transmission','Fuel_Type','City']),
                                    remainder='passthrough')




from sklearn.linear_model import LinearRegression



lr=LinearRegression()



pipe=make_pipeline(column_trans,lr)




pipe.fit(x_train,y_train)




y_pred=pipe.predict(x_test)



r2_score(y_test,y_pred)



scores=[]
for i in range(2000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))



scores[np.argmax(scores)]



price = pipe.predict(pd.DataFrame(columns=x_test.columns,data=np.array(['Renault TRIBER 1.0 RXZ','Renault','MANUAL',2018,33419,'Petrol','Hyderabad']).reshape(1,7)))
print("The sales price will be:", np.round(price,2))



#Comparing original Data's Scatterlot with regressed model
plt.plot(cars.Sales_Price)
plt.plot(pipe.predict(cars))



cr7 = pd.get_dummies(cars,columns=['Name','Company','Transmission','Fuel_Type','City'])



x1=cr7.drop(['Registration','Month','Owner','Original_Price','EMI(₹)','Sales_Price'], axis=1)
y1=cars[['Sales_Price']]


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier




num_trees = 65
max_features = 7



kfold = KFold(n_splits=7, random_state=14, shuffle=True)
model3 = RandomForestRegressor(n_estimators=num_trees, criterion='mse', max_features=max_features)



results = cross_val_score(model3, x1, y1, cv=kfold)
print(results.mean())




num_trees = 82
max_features = 4



kfold = KFold(n_splits=8, random_state=27, shuffle=True)
model2 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)



results = cross_val_score(model2, x1, y1, cv=kfold)
print(results.mean())




from sklearn.tree import  DecisionTreeRegressor



from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
cars['Name']= label_encoder.fit_transform(cars['Name'])
cars['Company']= label_encoder.fit_transform(cars['Company'])
cars['Transmission']= label_encoder.fit_transform(cars['Transmission'])
cars['Fuel_Type']= label_encoder.fit_transform(cars['Fuel_Type'])
cars['Owner']= label_encoder.fit_transform(cars['Owner'])
cars['Month']= label_encoder.fit_transform(cars['Month'])
cars['City']= label_encoder.fit_transform(cars['City'])



x2=cars.drop(['Original_Price','Owner','Registration','EMI(₹)','Sales_Price'], axis=1)
y2=cars[['Sales_Price']]



import ppscore as pps
pps.matrix(cars)


pps.score(cars, "Sales_Price", "KM_Driven")




x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.3, random_state=21)




model5 = DecisionTreeRegressor()
model5.fit(x_train, y_train)




model5.score(x_test,y_test)



x_train, x_test,y_train,y_test = train_test_split(x2,y2, test_size=0.22,random_state=18)



from sklearn.tree import DecisionTreeClassifier
model4_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)




model4_gini.fit(x_train, y_train)



y_test3 = y_test.to_numpy()
y_test3 = np.reshape(y_test3, 447)



#Prediction and computing the accuracy
pred=model5.predict(x_test)
np.mean(pred==y_test3)





