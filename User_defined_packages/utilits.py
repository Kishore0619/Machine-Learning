import pandas as pd
# load the user define dclass to import data
from User_defined_Data_loader import DataLoader
import pandas as pd
import numpy as np
import datetime as dt
pd.set_option('display.max_columns', None)

# import libraries for the Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as px
from plotly.subplots import make_subplots
import plotly.colors
# set the color palette
sns.set_palette('Pastel1')

from sklearn.preprocessing import MinMaxScaler as mms
import pickle
        
"""
--------------------------------------------------------------------------------------------------------------------------------------------
Function to handle Null values

"""        
def Handling_null_values(data,column_name,thresh_hold):
    if (data.isna().sum() > 0) and (data.isna().sum() < thresh_hold*len(data)):
        if data.dtype == 'int' or data.dtype == 'float':
            if  len(data[data>(data.mean()+(2*data.std()))]) > 0:
                filled_data = data.fillna(data.median())
                return filled_data
            else:
                filled_data = data.fillna(data.mean())
                return filled_data
        elif data.dtype == 'object':
            filled_data = data.fillna(data.mode())
            return filled_data
        else:
            print("Data is neither Numerical nor Categorical!.")
    elif data.isna().sum() > thresh_hold*len(data):
        print(f"Drop the column: {column_name}")
    else:
        return data
    
# funtion to visualize the numerical Variable types 
def univariate_plot_numerical(data,col_name):
    # initialize the figure with 2 subplots in 1 row
    fig = make_subplots(cols=2,rows=1,subplot_titles=(f'{col_name} - Box plot',f'{col_name} - Hist plot'))
    # set height and width of the figure
    fig.update_layout(autosize=False,width=1000,height=500)
    # add box plot to the figure
    fig.add_trace(px.Box(x=data,name=str(col_name)),row=1,col=1)
    # add Histogram plot to the figure
    fig.add_trace(px.Histogram(x=data,nbinsx=15,name=str(col_name)),row=1,col=2)
    # show the figure
    fig.show()
    
# function to visualize the Object type variables
def univariate_plot_object(data,col_name):
    
    # define a function to plot the data
    def plotting_function(plot_data):
        # add bar plot to the figure
        fig.add_trace(px.Bar(x=plot_data.value_counts().index.astype(str),
                        y=plot_data.value_counts().values,
                        text=plot_data.value_counts().values,
                        marker_color=plotly.colors.qualitative.Plotly
                        ),
                row=1,col=1)  
        # add pie chart to the figure
        fig.add_trace(px.Pie(labels=plot_data.value_counts().index.astype(str), 
                        values=plot_data.value_counts().values,
                        domain={'x': [0.5, 1]}))
        # to show the figure
        fig.show()
        
    # initialize the the figure with 2 subplots
    fig = make_subplots(cols=2,rows=1,
                        subplot_titles=(f'{col_name} - Count plot',f'{col_name} - Pie Chart')
                    )
    # update the height and width of the figure
    fig.update_layout(autosize=False,width=1000,height=500)
    
    # check if the object column has more than 10 unique values
    if len(data.unique())>10:
        # select the top 10 unique values form the column
        data_keys = data.value_counts().head(10).keys()
        # store the data of top 10 column values in temp variable
        new_data = data[data.isin(data_keys)]
        # call the plotting function
        plotting_function(new_data)
        
    # if column doesnt have the more than 10 unique values    
    else:
        # call the plotting function
        plotting_function(data)
        
        
import plotly.express as go
# creating function to plot numerical column with object column
def numerical_with_object_box_plot(data,numerical_column,category_column):
    fig = px.Figure()
    fig.update_layout(autosize=False,width=1000,height=500,xaxis_title=f'{category_column}', yaxis_title=f'{numerical_column}') 
    groups = data[category_column].value_counts().keys()[:10]
    colors = iter([
                    '#FFA500', '#800080', '#008000', '#000080',
                    '#A52A2A', '#808080','#FFD700', '#FF6347', 
                    '#808000', '#FF1493'
                ])
    for i in groups:
        fig.add_trace(px.Box(y=data.loc[data[category_column]==i,numerical_column],name=i,marker_color=next(colors),showlegend=False))
    fig.show()


    
def object_with_object_countplot(data, category_column1, category_column2):
    fig = px.Figure()
    colors = [
                    '#FFA500', '#800080', '#008000', '#000080',
                    '#A52A2A', '#808080','#FFD700', '#FF6347', 
                    '#808000', '#FF1493'
                ]
    fig.update_layout(autosize=False, width=1000, height=500,barmode='group',
                    xaxis_title=f'{category_column1}', yaxis_title='count')
    
    category_counts = data.groupby([category_column1, category_column2]).size().reset_index(name='count')

    for target_category in category_counts[category_column1].unique():
        subset = category_counts[category_counts[category_column2] == target_category]
        fig.add_trace(px.Bar(x=subset[category_column1].astype(str), y=subset['count'], name=str(target_category),text=subset['count'], textposition='auto'))

    fig.show()

    
    
            
# creating the function to plot numerical with numerical    
def numerical_with_numerical_scatterplot(data,numerical_column1,numerical_column2):
    fig,ax = plt.subplots(1,1,figsize=(7,4))
    sns.scatterplot(x=data[numerical_column1],y=data[numerical_column2])



import numpy as np
# defining the function to handle the outliers
def handling_Outliers(data):
    # 25 percentile of the data
    q1 = np.percentile(data,25)
    # 75 percentile of the data
    q3 = np.percentile(data,75)
    # Inter quatile range
    iqr = q3-q1
    # upper boundary 
    upper_boundary = q3+1.5*iqr
    # lower boundary
    lower_boundary = q1-1.5*iqr
    # replacing the values with nan which have greater the uper boudary value
    data[data>upper_boundary] = np.nan
    # replacing the values with nan whihc have less than the lower boundary
    data[data<lower_boundary] = np.nan
    return data




def data_reading(path):
    dl = DataLoader(path)
    validation_data = dl.read_data()
    validation_data.drop(['Churn','CustomerID'],axis=1,inplace=True)
    return validation_data


def preprocess_data(data):
    # convert duplicate entries to original entries
    data['PreferredPaymentMode'] = data['PreferredPaymentMode'].replace({'COD':'Cash on Delivery','CC':'Credit Card'})
    data['PreferredLoginDevice'] = data['PreferredLoginDevice'].replace({'Phone':'Mobile Phone'})
    data['PreferedOrderCat'] = data['PreferedOrderCat'].replace({'Mobile':'Mobile Phone'})

    # convert the discrete values to nominal
    data['Tenure'] = data['Tenure'].apply(lambda x:'>21 years' if x>21 else '16-20 years' if x>15 else '8-15 years' if x>7 else '0-7 years')
    data['WarehouseToHome'] = data['WarehouseToHome'].apply(lambda x:'>40 km' if x>40 else '20-39 km' if x>20 else '<20 km')
    data['NumberOfAddress'] = data['NumberOfAddress'].apply(lambda x: '>10' if x>10 else '6-10' if x>5 else '0-5')
    data['OrderAmountHikeFromlastYear'] = data['OrderAmountHikeFromlastYear'].apply(lambda x: '> 20%' if x>20 else 'Between 16-20%' if x>15 else 'upto 15%')
    data['DaySinceLastOrder'] = data['DaySinceLastOrder'].apply(lambda x:'>10 days' if x>10 else '6-10 days' if x>5 else '0-5 days')

    return data


def encoding_and_transfrom(data):
    # encoding the data
    data['Tenure'] = data['Tenure'].replace({'0-7 years':0,'8-15 years':2,'16-20 years':1,'>21 years':3})
    data['WarehouseToHome'] = data['WarehouseToHome'].replace({'<20 km':1,'20-39 km':0,'>40 km':2})
    data['NumberOfAddress'] = data['NumberOfAddress'].replace({'0-5':0,'6-10':1,'>10':86})
    data['OrderAmountHikeFromlastYear'] = data['OrderAmountHikeFromlastYear'].replace({'upto 15%':2,'Between 16-20%':1,'> 20%':0})
    data['DaySinceLastOrder'] = data['DaySinceLastOrder'].replace({'0-5 days':0,'6-10 days':1,'>10 days':2})
    data['PreferredLoginDevice']=data['PreferredLoginDevice'].replace({'Mobile Phone':1,'Computer':0})
    data['PreferredPaymentMode'] = data['PreferredPaymentMode'].replace({'Debit Card':2,'Credit Card':1,'E wallet':3,'Cash on Delivery':0,'UPI':4})
    data['Gender'] = data['Gender'].replace({'Male':1,'Female':0})
    data['PreferedOrderCat'] = data['PreferedOrderCat'].replace({'Mobile Phone':3,'Laptop & Accessory':2,'Fashion':0,'Grocery':1,'Others':4})
    data['MaritalStatus'] = data['MaritalStatus'].replace({'Married':1,'Single':2,'Divorced':0})

    # scale to data
    scaled_data = mms.transform(data)

    return scaled_data


def model_prediction(scaled_data):
    pickle_model = open("Random_forest.pickle" , "rb")
    random_forest_model = pickle.load(pickle_model)
    predictions = random_forest_model.predict(scaled_data)
    return predictions


def prediction_pipeline(path):
    validation_data = data_reading(path)
    cleaned_data = preprocess_data(validation_data)
    scaled_data = encoding_and_transfrom(cleaned_data)
    prediction_values = model_prediction(scaled_data)
    return prediction_values