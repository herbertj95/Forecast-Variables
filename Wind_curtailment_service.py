# -*- coding: utf-8 -*-
"""
Created on July 16 of 2023
Wind Curtailment Service Forecast for EV4EU
@author: Herbert Amezquita
"""
###############################################################################################################################
'Libraries'
###############################################################################################################################
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import xgboost as xgb
from xgboost import plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report, accuracy_score
from sklearn import metrics
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
import itertools

###############################################################################################################################
'Plot Parameters'
###############################################################################################################################
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20})
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

###############################################################################################################################
'Functions'
###############################################################################################################################
'Function to define the season based on the month'
def define_season(month_number):
    if month_number in [12,1,2]:
        return 1
    elif month_number in [3,4,5]:
        return 2
    elif month_number in [6,7,8]:
        return 3
    elif month_number in [9,10,11]:
        return 4

'Function create_features'
def create_features(df):
    """
    Creates date/time features from a dataframe 
    
    Args:
        df - dataframe with a datetime index
        
    Returns:
        df - dataframe with 'Weekofyear','Dayofyear','Month','Dayofmonth',
             'Dayofweek','Weekend','Season','Holiday','Hour' and 'Minute' features created
    """
    
    df['Date'] = df.index
    df['Weekofyear'] = df['Date'].dt.weekofyear   #Value: 1-52
    df['Dayofyear'] = df['Date'].dt.dayofyear    #Value: 1-365
    df['Month'] = df['Date'].dt.month   #Value: 1-12
    df['Dayofmonth'] = df['Date'].dt.day   #Value: 1-30/31
    df['Dayofweek']= df['Date'].dt.weekday+1     #Value: 1-7 (Monday-Sunday)
    df['Weekend']= np.where((df['Dayofweek']==6) | (df['Dayofweek']==7), 1, 0)    #Value: 1 if weekend, 0 if not
    df['Season']= df.Month.apply(define_season)    #Value 1-4 (winter, spring, summer and fall)    
    df['Hour'] = df['Date'].dt.hour
    df['Hour']= (df['Hour']+24).where(df['Hour']==0, df['Hour'])    #Value: 1-24
    df['Minute']= df['Date'].dt.minute     #Value: 0, 15, 30 or 45
    df= df.drop(['Date'], axis=1)
    
    return df

'Function lag_features'
def lag_features(lag_dataset, days_list, var):
    
    temp_data = lag_dataset[var]
    
    for days in days_list:
        rows = 96 * days
        lag_dataset[var + "_lag_{}".format(days)] = temp_data.shift(rows)

    return lag_dataset 

'Function cyclical_features'
def cyclical_features(df):
    """
    Transforms (date/time) features into cyclical sine and cosine features
    
    Args:
        df - dataframe with 'Weekofyear','Dayofyear','Season','Month',
             'Dayofmonth','Dayofweek','Hour','Minute' columns
        
    Returns:
        df - dataframe including the cyclical features (x and y for each column)
    """
    
    df['Weekofyear_x']= np.cos(df['Weekofyear']*2*np.pi/52)
    df['Weekofyear_y']= np.sin(df['Weekofyear']*2*np.pi/52)
    df['Dayofyear_x']= np.cos(df['Dayofyear']*2*np.pi/365)
    df['Dayofyear_y']= np.sin(df['Dayofyear']*2*np.pi/365)
    df['Season_x']= np.cos(df['Season']*2*np.pi/4)
    df['Season_y']= np.sin(df['Season']*2*np.pi/4)
    df['Month_x']= np.cos(df['Month']*2*np.pi/12)
    df['Month_y']= np.sin(df['Month']*2*np.pi/12)
    df['Dayofmonth_x']= np.cos(df['Dayofmonth']*2*np.pi/31)
    df['Dayofmonth_y']= np.sin(df['Dayofmonth']*2*np.pi/31)
    df['Dayofweek_x']= np.cos(df['Dayofweek']*2*np.pi/7)
    df['Dayofweek_y']= np.sin(df['Dayofweek']*2*np.pi/7)
    df['Hour_x']= np.cos(df['Hour']*2*np.pi/24)
    df['Hour_y']= np.sin(df['Hour']*2*np.pi/24)
    df['Minute_x']= np.cos(df['Minute']*2*np.pi/45)
    df['Minute_y']= np.sin(df['Minute']*2*np.pi/45)
    df= df.drop(columns=['Weekofyear','Dayofyear','Season','Month','Dayofmonth',
                                         'Dayofweek','Hour','Minute'])
    
    return df

'Function plot_confusion_matrix'
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

###############################################################################################################################
'############################################Wind Curtailment Service Forecast#################################################'
###############################################################################################################################

def forecast(var, season, start_forecast):
    print('Forecast variable: ', var)
    
    'Reading services data'
    raw_data = pd.read_csv('./Datasets/Services Data.csv', parse_dates= ['Date'])
    raw_data.set_index('Date', inplace= True)
    raw_data.rename(columns = {'Curtailed Power (kW)': 'Wind curt'}, inplace = True)
    
    data_final = raw_data.loc[:, ['Wind curt']].copy()
    
    'Creating date/time features using datetime column Date as index'
    data_final = create_features(data_final)
     
    'Creating lag features of 1 day, 5 days and 1 week before'
    data_final = lag_features(data_final,[1,5,7], 'Wind curt')
    data_final.fillna(0, inplace= True)
    data_final.dropna(inplace=True)
    
    'Transforming date/time features into two dimensional features'
    data_final= cyclical_features(data_final)
    
    'Array containing the names of all features available'
    all_features = data_final.columns.values.tolist()
    all_features.remove('Wind curt')
    all_features= np.array(all_features) 
    
    X = data_final.values
    Y = X[:,0] 
    X = X[:,[x for x in range(1,len(all_features)+1)]]
    
    'Feature selection for the model'
    reg_RF= RandomForestRegressor()
    reg_RF.fit(X, Y)
    importance= pd.DataFrame(data= {'Feature': all_features, 'Score': reg_RF.feature_importances_})
    importance= importance.sort_values(by=['Score'], ascending= False)
    importance.set_index('Feature', inplace=True)
    
    'Defining the number of features to use in the models'
    num_features = 15   #Optimal number of features is 15
    
    'Defining training and test periods'
    data_train = data_final.loc[: start_forecast - timedelta(minutes= 15)]
    data_test = data_final.loc[start_forecast : start_forecast + relativedelta(months= 1) - timedelta(minutes= 15)]
    
    'Plot train-test '
    fig,ax = plt.subplots()
    coloring = data_final['Wind curt'].max()
    plt.plot(data_train.index, data_train['Wind curt'], color= "darkcyan", alpha= 0.75)
    plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
    plt.plot(data_test.index, data_test['Wind curt'], color = "dodgerblue", alpha= 0.60)
    plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("Power (kW)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(['Train','Test'], frameon= False, loc= 'upper center', ncol= 2)
    plt.title(season+ " Train - Test split for Wind Curtailment Service", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Features used'
    USE_COLUMNS = importance[:num_features].index.values
    
    FORECAST_COLUMN = ['Wind curt']
    
    print('The features used in the RF model are:', USE_COLUMNS)
    
    'RF model'
    xtrain = data_train.loc[:, USE_COLUMNS]
    xtest = data_test.loc[:, USE_COLUMNS]
    ytrain = data_train.loc[:, FORECAST_COLUMN]
    ytest = data_test.loc[:, FORECAST_COLUMN]
    
    reg_RF.fit(xtrain, np.ravel(ytrain))
    
    'Predictions'
    df_RF = pd.DataFrame(reg_RF.predict(xtest), columns= ['Prediction'], index= xtest.index)
    df_RF['Real']= ytest
    
    'Plots'
    #Regression Plot
    sns.scatterplot(data= df_RF, x='Real', y= 'Prediction')
    plt.plot(ytest, ytest, color = "dodgerblue", linewidth= 2) 
    plt.xlabel("Real Power (kW)", alpha= 0.75, weight= "bold")
    plt.ylabel("Predicted Power (kW)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold",fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.title(season + " correlation real vs predictions for Wind Curtailment Service", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    #Real vs predictions in the same plot
    fig,ax = plt.subplots()
    ax.plot(df_RF.Real, label= "Real")
    ax.plot(df_RF.Prediction, label= "Predicted", ls= '--')
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("Power (kW)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold",fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(frameon= False, loc= 'best')
    plt.title(season + " real vs predicted for Wind Curtailment Service", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Errors'
    MAE_RF = metrics.mean_absolute_error(df_RF.Real, df_RF.Prediction)
    RMSE_RF = np.sqrt(metrics.mean_squared_error(df_RF.Real, df_RF.Prediction))
    normRMSE_RF = 100 * RMSE_RF / ytest['Wind curt'].max()
    R2_RF = metrics.r2_score(df_RF.Real, df_RF.Prediction)
    
    print('RF - Mean Absolute Error (MAE):', round(MAE_RF,2))
    print('RF - Root Mean Square Error (RMSE):',  round(RMSE_RF,2))
    print('RF - Normalized RMSE (%):', round(normRMSE_RF,2))
    print('RF - R square (%):', round(R2_RF,2))
    
    'Post-Processing'
    vehicles= 500
    df_RF[var]= np.where((df_RF.Prediction > (vehicles * 3.7)), 1, 0)
    df_RF[var]= np.where((df_RF.index.hour > 6) | (df_RF.index.hour < 0), 0, df_RF[var])
    df_RF['Wind Curtailment'] = raw_data['Wind Curtailment Service Activation']
    
    #Accuracy
    accuracy = accuracy_score(df_RF['Wind Curtailment'],  df_RF[var])
    print(f'Accuracy for {var}: {accuracy:.2f}')
    
    #Confusion matrix
    cnf_matrix = confusion_matrix(df_RF['Wind Curtailment'],  df_RF[var], labels=[1,0])
    np.set_printoptions(precision=2)
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes= ['Activation= 1','Activation= 0'], normalize= False, title= season + ' confusion matrix for '+ var)
    plt.show()
    
    'Forecast result'
    predictions_wind_curtailment = df_RF.loc[:, [var]]
    
    return predictions_wind_curtailment
    