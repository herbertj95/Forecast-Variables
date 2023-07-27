# -*- coding: utf-8 -*-
"""
Created on July 12 of 2023
Load Forecast for EV4EU
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
from sklearn import metrics
from datetime import datetime
from datetime import timedelta
import warnings

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

###############################################################################################################################
'##########################################Load (Building Consumption) Forecast################################################'
###############################################################################################################################

def forecast(data, var, season, start_forecast):
    
    print('Forecast variable: ', var)
    data_final = data.loc[:, [var]].copy()
        
    'Reading meteo data Solcast'
    data_meteo = pd.read_csv('./Datasets/Meteo Solcast.csv')
    data_meteo['Date'] = pd.to_datetime(data_meteo.PeriodStart)
    data_meteo['Date'] = data_meteo['Date'].dt.tz_localize(None)
    data_meteo.set_index('Date', inplace= True)
    data_meteo = data_meteo.drop(['PeriodStart','PeriodEnd'], axis= 1)
    
    'Merging raw data with meteo data'
    data_final = data_final.join(data_meteo)
    
    'Creating date/time features using datetime column Date as index'
    data_final = create_features(data_final)
    
    'Creating lag features of 1 day, 5 days and 1 week before'
    data_final = lag_features(data_final,[1,5,7], var)
    data_final.fillna(0, inplace= True)
    data_final.dropna(inplace=True)
    
    'Barplot average energy consumption per hour'
    # mean_per_hour = data_final.groupby('Hour')[var].agg(["mean"])
    # fig, ax = plt.subplots()
    # plt.bar(mean_per_hour.index, mean_per_hour["mean"], color = 'darkcyan')
    # plt.xticks(range(1,25), alpha=0.75, weight= "bold")
    # plt.yticks(alpha= 0.75, weight= "bold")
    # plt.xlabel("Hour", alpha=0.75, weight= "bold")
    # plt.ylabel("Energy (kWh)", alpha= 0.75, weight= "bold")
    # plt.title("Average load per hour", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
    # plt.show()
    
    'Transforming date/time features into two dimensional features'
    data_final = cyclical_features(data_final)
    
    'Array containing the names of all features available'
    all_features = data_final.columns.values.tolist()
    all_features.remove(var)
    all_features= np.array(all_features) 
    
    X = data_final.values
    Y = X[:,0] 
    X = X[:,[x for x in range(1,len(all_features)+1)]]
    
    'Feature selection for the model'
    parameters_XGBOOST = {'n_estimators' : 500,
                      'learning_rate' : 0.01,
                      'verbosity' : 0,
                      'n_jobs' : -1,
                      'gamma' : 0,
                      'min_child_weight' : 1,
                      'max_delta_step' : 0,
                      'subsample' : 0.7,
                      'colsample_bytree' : 1,
                      'colsample_bylevel' : 1,
                      'colsample_bynode' : 1,
                      'reg_alpha' : 0,
                      'reg_lambda' : 1,
                      'random_state' : 18,
                      'objective' : 'reg:linear',
                      'booster' : 'gbtree'}
    
    reg_XGBOOST = xgb.XGBRegressor(**parameters_XGBOOST)
    reg_XGBOOST.fit(X, Y)
    importance = pd.DataFrame(data= {'Feature': all_features, 'Score': reg_XGBOOST.feature_importances_})
    importance = importance.sort_values(by= ['Score'], ascending= False)
    importance.set_index('Feature', inplace= True)
    
    'Defining the number of features to use in the models'
    num_features = 20   #Optimal number of features is 20
    
    'Defining training and test periods'
    data_train = data_final.loc[: start_forecast - timedelta(minutes= 15)]
    data_test = data_final.loc[start_forecast :]
    
    'Plot train-test '
    fig,ax = plt.subplots()
    coloring = data_final[var].max()
    plt.plot(data_train.index, data_train[var], color= "darkcyan", alpha= 0.75)
    plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
    plt.plot(data_test.index, data_test[var], color = "dodgerblue", alpha= 0.60)
    plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("Energy (kWh)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(['Train','Test'], frameon= False, loc= 'upper center', ncol= 2)
    plt.title(season+ " Train - Test split for "+ var, alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Features used'
    USE_COLUMNS = importance[:num_features].index.values
    
    FORECAST_COLUMN = [var]
    
    print('The features used in the XGBOOST model are:', USE_COLUMNS)
    
    'XGBOOST model'
    xtrain = data_train.loc[:, USE_COLUMNS]
    xtest = data_test.loc[:, USE_COLUMNS]
    ytrain = data_train.loc[:, FORECAST_COLUMN]
    ytest = data_test.loc[:, FORECAST_COLUMN]
    
    reg_XGBOOST.fit(xtrain, np.ravel(ytrain))
    
    'Predictions and pos-processing'
    df_XGBOOST = pd.DataFrame(reg_XGBOOST.predict(xtest), columns= ['Prediction'], index= xtest.index)
    df_XGBOOST['Prediction']= np.where(df_XGBOOST['Prediction']< 0, 0 , df_XGBOOST['Prediction'])
    df_XGBOOST['Real'] = ytest
    
    'Plots'
    #Regression Plot
    sns.scatterplot(data= df_XGBOOST, x='Real', y= 'Prediction')
    plt.plot(ytest, ytest, color = "dodgerblue", linewidth= 2) 
    plt.xlabel("Real consumption (kWh)", alpha= 0.75, weight= "bold")
    plt.ylabel("Predicted consumption (kWh)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold",fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.title(season + " correlation real vs predictions for "+ var, alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    #Real vs predictions in the same plot
    fig,ax = plt.subplots()
    ax.plot(df_XGBOOST.Real, label= "Real")
    ax.plot(df_XGBOOST.Prediction, label= "Predicted", ls= '--')
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("Energy (kWh)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold",fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(frameon= False, loc= 'best')
    plt.title(season + " real vs predicted for "+ var, alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Errors'
    MAE_XGBOOST = metrics.mean_absolute_error(df_XGBOOST.Real, df_XGBOOST.Prediction)
    RMSE_XGBOOST = np.sqrt(metrics.mean_squared_error(df_XGBOOST.Real, df_XGBOOST.Prediction))
    normRMSE_XGBOOST = 100 * RMSE_XGBOOST / ytest[var].max()
    R2_XGBOOST = metrics.r2_score(df_XGBOOST.Real, df_XGBOOST.Prediction)
    
    print('XGBOOST- Mean Absolute Error (MAE):', round(MAE_XGBOOST,2))
    print('XGBOOST - Root Mean Square Error (RMSE):',  round(RMSE_XGBOOST,2))
    print('XGBOOST - Normalized RMSE (%):', round(normRMSE_XGBOOST,2))
    #print('XGBOOST - R square (%):', round(R2_XGBOOST,2))
    
    if var == 'build cons #1 (kWh)' or var == 'build cons #2 (kWh)':
        print('#################################################################')
    
    ###############################################################################################################################
    'Forecast results'
    ###############################################################################################################################
    
    predictions_load = df_XGBOOST.Prediction
    predictions_load.rename(var, inplace= True)
    
    return predictions_load
