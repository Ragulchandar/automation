from heapq import merge
from telnetlib import GA
from turtle import color
from sklearn.feature_selection import SelectorMixin
import streamlit as st
import math
import altair as alt

st.title('Metalmark ML Automation')

import pandas as pd
import numpy as np
from io import StringIO
from pandas.core.arrays.datetimelike import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix
from sklearn import neighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.header('Upload your files')
uploaded_file_1 = st.file_uploader("1. Choose a file (csv)",key=2)
if uploaded_file_1 is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file_1)
    st.write(dataframe)

uploaded_file_2 = st.file_uploader("1. Choose a file (csv)",key=3)
if uploaded_file_2 is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file_2)
    st.write(dataframe)
def say_hello(name):
    st.write("Hello " + name)
    

def mergeDf(df1,df2):
#  df1 = pd.read_csv(file_name_1)
#  df2 = pd.read_csv(file_name_2)
# set time as index
# in this case we are using 'time' column
# but in your case it might be called something else
# make sure to change it accordingy
 df1.astype(str)
 df1["DateAndTime"] = df1["Date"] + " " +df1["Time"]
 df1["DateAndTime"] = pd.to_datetime(df1["DateAndTime"])
 df2.astype(str)
 df2["DateAndTime"] = df2["Date"] + " " +df2["Time"]
 df2["DateAndTime"] = pd.to_datetime(df2["DateAndTime"])

 df1.set_index('DateAndTime', inplace=True)
 df2.set_index('DateAndTime', inplace=True)

# identify the time difference in df1
 interval_1 = df1.index[1] - df1.index[0]
 secs_1 = interval_1.total_seconds()
# identify the time difference in df2
 interval_2 = df2.index[1] - df2.index[0]
 secs_2 = interval_2.total_seconds()

# the LCM of the two intervals is the new time interval we want
# new interval lcm of two float
 def lcm(a, b):
     return abs(a*b) // math.gcd(a, b) if a and b else 0
 def lcm_float(a, b, precision):
    a = round(a, precision)
    b = round(b, precision)
    return math.lcm(int(a*10**precision), int(b*10**precision))/10**precision
 new_interval = lcm_float(secs_1, secs_2, 2)
 new_interval= timedelta(seconds=20)

# resample the dataframes to the new interval
# here we use mean() to aggregate the resampled data
 resampled_df1 = df1.resample(new_interval).mean()
 resampled_df2 = df2.resample(new_interval).mean()

# perform a inner join on the two dataframes
 out_df = pd.merge(resampled_df1, resampled_df2, how='inner', left_index=True, right_index=True)

 return out_df

merge= st.button("merge")
if merge:
    if uploaded_file_2 is not None and uploaded_file_1 is not None:
        # st.write(uploaded_file_1)
        # st.write(uploaded_file_2)
        uploaded_file_1.seek(0)
        uploaded_file_2.seek(0)
        df_1= pd.read_csv(uploaded_file_1)
        df_2= pd.read_csv(uploaded_file_2)
        out_df=mergeDf(df_1,df_2)
        st.write(out_df)
        # st write number of rows
        st.write("Number of rows: ", out_df.shape[0])
        st.write("Number of null values in each column:", out_df.isna().sum())
        

def clean_col(df):
 df= df.dropna(axis=1, how='any', thresh=df.shape[0]*0.4)
 cols = df.columns
 for i in cols :
  if df[i].isna().sum() < df.shape[0]*0.4:
   df[i] = df[i].fillna(df[i].mean())
  else:
    df = df.drop(columns=i)
    
 return df


clean= st.button("Clean")
if clean:
    if uploaded_file_2 is not None and uploaded_file_1 is not None:
        uploaded_file_1.seek(0)
        uploaded_file_2.seek(0)
        df_1= pd.read_csv(uploaded_file_1)
        df_2= pd.read_csv(uploaded_file_2)
        out_df=mergeDf(df_1,df_2)
        clean_df=clean_col(out_df)
        #st.write(clean_df)
        st.write("Number of null values in each column:", clean_df.isna().sum())
        st.write("Number of rows: ", clean_df.shape[0])

st.header('Exploratory Data Analysis')


def cor(df):
    corr = df.corr()
    return corr

corr = st.button("Correlation")
if corr:
    if uploaded_file_2 is not None and uploaded_file_1 is not None:
        uploaded_file_1.seek(0)
        uploaded_file_2.seek(0)
        df_1= pd.read_csv(uploaded_file_1)
        df_2= pd.read_csv(uploaded_file_2)
        out_df=mergeDf(df_1,df_2)
        clean_df=clean_col(out_df)
        corr_df=cor(clean_df)
        st.write(corr_df)


#def pairplot(df):
  
    #st.write(df[[Gases,Sensors]])


Gases= st.multiselect('Select Gases', options=["CO2_35C", "Acetaldehyde__35c", "Formaldehyde_35c"], key=9)
Sensors= st.multiselect('Select Sensors', options=['VOC_Sensor_{device="MQ135",_name="VOC_data"}', 'VOC_Sensor_{device="MQ138",_name="VOC_data"}', 'VOC_Sensor_{device="MQ9",_name="VOC_data"}', 'VOC_Sensor_{device="MQ3",_name="VOC_data"}'], key=10)
pairplot = st.button("Pairplot")
if pairplot:
    if uploaded_file_2 is not None and uploaded_file_1 is not None:
        uploaded_file_1.seek(0)
        uploaded_file_2.seek(0)
        df_1= pd.read_csv(uploaded_file_1)
        df_2= pd.read_csv(uploaded_file_2)
        out_df=mergeDf(df_1,df_2)
        clean_df=clean_col(out_df)
        clean_df=clean_df.rename(columns=lambda x:x.replace(' ','_'))
        #print columns
        # st.write(clean_df.columns)
        # print(clean_df[Gases]) 
        # df_plt= clean_df[[Gases, Sensors]]
        #corr_df=cor(clean_df)



        # plot pair wise scatter plot for Gases and Sensors
        # plots in different subplots
        # subplots of leng(Gases) * len(Sensors)
        print(len(Gases), len(Sensors))
        # fisize adpat to number of subplots
        fig, axs = plt.subplots(len(Gases), len(Sensors), figsize=(len(Sensors)*5, len(Gases)*5))
        # get shape of axs
        if len(Gases) > 1 or len(Sensors) > 1:  
            axs_shape = axs.shape
            # if axs is 1D array
            if len(axs_shape) == 1:
                for i in range(len(Gases)):
                    for j in range(len(Sensors)):
                        axs[j].scatter(clean_df[Gases[i]], clean_df[Sensors[j]])
                        axs[j].set_xlabel(Gases[i])
                        axs[j].set_ylabel(Sensors[j])
            # if axs is 2D array
            elif len(axs_shape) == 2:
                for i in range(len(Gases)):
                    for j in range(len(Sensors)):
                        axs[i][j].scatter(clean_df[Gases[i]], clean_df[Sensors[j]])
                        axs[i][j].set_xlabel(Gases[i])
                        axs[i][j].set_ylabel(Sensors[j])
        # if axs is not 1D or 2D array
        else:
            axs.scatter(clean_df[Gases[0]], clean_df[Sensors[0]])
            axs.set_xlabel(Gases[0])
            axs.set_ylabel(Sensors[0])
        st.pyplot(fig)
       

Gases= st.selectbox('Select Gases', options=['CO2 35C', 'Acetaldehyde  35c', 'Formaldehyde 35c'], key='1')
Sensors= st.selectbox('Select Sensors', options=['VOC_Sensor {device="MQ135", name="VOC_data"}', 'VOC_Sensor {device="MQ138", name="VOC_data"}', 'VOC_Sensor {device="MQ9", name="VOC_data"}', 'VOC_Sensor {device="MQ3", name="VOC_data"}'], key='4')
lineplot = st.button("Lineplot")
if lineplot:
    if uploaded_file_2 is not None and uploaded_file_1 is not None:
        uploaded_file_1.seek(0)
        uploaded_file_2.seek(0)
        df_1= pd.read_csv(uploaded_file_1)
        df_2= pd.read_csv(uploaded_file_2)
        out_df=mergeDf(df_1,df_2)
        clean_df=clean_col(out_df)
        df_plt= clean_df[[Gases, Sensors]]
        
        #corr_df=cor(clean_df)
        fig, ax = plt.subplots()
        fig = st.line_chart(df_plt, height=500)

st.header('Machine Learning Models')

Gases= st.selectbox('Select Gas for Prediction', options=['CO2 35C', 'Acetaldehyde  35c', 'Formaldehyde 35c'], key = '5')
def Split_train_test(df):
    X = df.drop(columns = Gases).copy()
    y = df[Gases].copy()

    # Splitting 80% training and 20% validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state = 1)
    st.write("The shape of each data:\n")
    st.write(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def scaling(X_train, X_test):
    scaler = MinMaxScaler()
    #Training data
    X_num_train_scale=scaler.fit_transform(X_train)
    X_num_train_scale = pd.DataFrame(X_num_train_scale)
    X_num_train_scale.columns = X_train.columns
    X_num_train_scale.index = X_train.index
    X_train = X_num_train_scale

    # Testing data
    X_num_test_scale = scaler.transform(X_test)
    X_num_test_scale = pd.DataFrame(X_num_test_scale)
    X_num_test_scale.columns = X_test.columns
    X_num_test_scale.index = X_test.index
    X_test = X_num_test_scale

    return X_train, X_test

def l_regression(df):
    X_train, X_test, y_train, y_test = Split_train_test(df)
    X_train, X_test = scaling(X_train, X_test)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    lr_score = r2_score(y_test, y_pred)
    lr_rmse = mean_squared_error(y_test, y_pred, squared = False)
    #st.write("Coefficients: ", model.coef_)
    #st.write("Intercept: ", model.intercept_)
    st.write("R2 Score: ", lr_score)
    st.write("RMSE: ", lr_rmse)

st.header('Linear Regression Machine Learning Models')

L_Regression = st.button("Linear Regression")
if L_Regression:
    if uploaded_file_2 is not None and uploaded_file_1 is not None:
        uploaded_file_1.seek(0)
        uploaded_file_2.seek(0)
        df_1= pd.read_csv(uploaded_file_1)
        df_2= pd.read_csv(uploaded_file_2)
        out_df=mergeDf(df_1,df_2)
        clean_df=clean_col(out_df)
        l_regression(clean_df)

st.header('KNN Regression Machine Learning Model')

def knn_reg(df):
    X_train, X_test, y_train, y_test = Split_train_test(df)
    X_train, X_test = scaling(X_train, X_test)
    knn = neighbors.KNeighborsRegressor(n_neighbors = 2)
    knn.fit(X_train, y_train)  #fit the model
    y_pred=knn.predict(X_test) 

    dtr_score = r2_score(y_test, y_pred)
    dtr_rmse = mean_squared_error(y_test, y_pred, squared = False)
    st.write("R2 Score: ", dtr_score)
    st.write("RMSE: ", dtr_rmse)

KNN = st.button("KNN")
if KNN:
    if uploaded_file_2 is not None and uploaded_file_1 is not None:
        uploaded_file_1.seek(0)
        uploaded_file_2.seek(0)
        df_1= pd.read_csv(uploaded_file_1)
        df_2= pd.read_csv(uploaded_file_2)
        out_df=mergeDf(df_1,df_2)
        clean_df=clean_col(out_df)
        knn_reg(clean_df)

st.header('Random Forest Regressor Machine Learning Model')

def rf_reg(df):
    X_train, X_test, y_train, y_test = Split_train_test(df)
    X_train, X_test = scaling(X_train, X_test)
    rfr = RandomForestRegressor().fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
  
    rfr_score = r2_score(y_test, y_pred)
    rfr_rmse = mean_squared_error(y_test, y_pred, squared = False)
    st.write("R2 Score: ", rfr_score)
    st.write("RMSE: ", rfr_rmse)

RF = st.button("Random Forest")
if RF:
    if uploaded_file_2 is not None and uploaded_file_1 is not None:
        uploaded_file_1.seek(0)
        uploaded_file_2.seek(0)
        df_1= pd.read_csv(uploaded_file_1)
        df_2= pd.read_csv(uploaded_file_2)
        out_df=mergeDf(df_1,df_2)
        clean_df=clean_col(out_df)
        rf_reg(clean_df)


st.header('All Regression Machine Learning Models')
Regression=st.button("Regression")
if Regression:
    if uploaded_file_2 is not None and uploaded_file_1 is not None:
        uploaded_file_1.seek(0)
        uploaded_file_2.seek(0)
        df_1= pd.read_csv(uploaded_file_1)
        df_2= pd.read_csv(uploaded_file_2)
        out_df=mergeDf(df_1,df_2)
        clean_df=clean_col(out_df)
        st.write("**Linear Regression**")
        l_regression(clean_df)
        st.write("**KNN Regression**")
        knn_reg(clean_df)
        st.write("**Random Forest Regression**")
        rf_reg(clean_df)

# # Brush for selection
# brush = alt.selection(type='interval')

# clean_clean_headers = clean_df.columns.tolist()
# # remove { and } from headers
# clean_clean_headers = [x.replace('{', '') for x in clean_clean_headers]
# clean_clean_headers = [x.replace('}', '') for x in clean_clean_headers]
# # remove " from headers
# clean_clean_headers = [x.replace('"', '') for x in clean_clean_headers]
# #rename dict 
# cleaned_df = clean_df.rename(columns=dict(zip(clean_df.columns, clean_clean_headers)))
# st.write(cleaned_df.columns.tolist())

# # Scatter Plot
# points = alt.Chart(cleaned_df).mark_point().encode(
#     x= alt.X('CO2 35C', type='quantitative'),
#     y= alt.Y('VOC_Sensor device=MQ135, name=VOC_data', type='quantitative'),
# ).add_selection(brush)

# # get extreme values of x and y from selection
# points_ =  alt.Chart(cleaned_df).mark_text().encode(
#     y=alt.Y('CO2 35C:Q',axis=None)
# ).transform_window(
#     row_number='max(CO2 35C:Q)'
# ).transform_filter(
#     brush
# ).transform_window(
#     rank='rank(row_number)'
# )

# horsepower = points_.encode(text='CO2 35C:N').properties(title='CO2 35C')

# ch = alt.hconcat(points, horsepower)

# st.altair_chart(ch, use_container_width=True)

# # get selected data and save it
# selected = points.transform_filter(brush)
# selected_df = selected.data


# # convert file to csv

# st.download_button("Download CSV", selected_df.to_csv(), file_name="selected_data.csv")

# get the highest adjusted r2 score for different degrees of polynomial features:
# def get_best_degree(df):
#     X_train, X_test, y_train, y_test = Split_train_test(df)
#     X_train, X_test = scaling(X_train, X_test)
#     r2_scores = []
#     for i in range(1, 10):
#         poly = PolynomialFeatures(degree=i)
#         X_train_poly = poly.fit_transform(X_train)
#         X_test_poly = poly.fit_transform(X_test)
#         poly.fit(X_train_poly, y_train)
#         lin2 = LinearRegression() 
#         lin2.fit(X_train_poly, y_train)
#         y_pred = lin2.predict(X_test_poly)













