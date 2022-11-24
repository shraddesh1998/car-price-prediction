import pandas as pd 
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title= "CARS24 DASHBOARD" ,
                page_icon= ":Plots:" ,
                layout= "wide")


cars = pd.read_csv("C:/Users/sansk/Downloads/cars_new_1.csv")

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

fig_1 = px.bar(
    cars , x = "Discounted_Price", y = "Fuel_Type",
    orientation="h",
    title="<b>Fuel_wise_sellingPrice</b>",

    color_discrete_sequence=["#0083B8"]*len(cars)
)



fig = px.scatter (cars , x= "Discounted_Price" , y = "EMI" , color= "Car_Type", symbol= "Fuel_Type"
, title="<b>Correlation between Variables</b>")

st.plotly_chart(fig)

fig_2 = px.pie(cars , values = "Discounted_Price" ,names = "Owner", title= "<b>Contribution of owner type</b>")


fig_3 = px.box(cars ,x='Car_Type',y='Discounted_Price',  title= "<b>Cars_Type and Price</b>")
st.plotly_chart(fig_3)

fig_4 = px.funnel(cars , x = "Discounted_Price" , y = "Car_Type" , title = "<b>Cars_types and price</b>")
st.plotly_chart(fig_4)

fig_5= px.histogram(cars , x = "Ragistration_year" , y = "Discounted_Price",
                 title= "year vs price")
st.plotly_chart(fig_5)                 

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_1 , use_container_width= True)
right_column.plotly_chart(fig_2 , use_container_width= True)

# ----------- HIDE STREAMLIT STYLE--------------
hide_st_style = """ 
               <style>
               #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
                </style>
                """
st.markdown(hide_st_style , unsafe_allow_html = True)               