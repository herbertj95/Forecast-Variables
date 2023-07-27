# -*- coding: utf-8 -*-
"""
Created on July 12 of 2023
Congestion Service Forecast for EV4EU
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
'##############################################Congestion Service Forecast####################################################'
###############################################################################################################################

def forecast(data, variables, season, start_forecast):
        
    print(f'Forecast variable: PT capacity')
    
    'Reading services data'
    raw_data = pd.read_csv('./Datasets/Services Data.csv', parse_dates= ['Date'])
    raw_data.set_index('Date', inplace= True)
    raw_data.rename(columns = {'PT capacity (-)': 'Congestion'}, inplace = True)
    
    data_final = raw_data.loc[:, ['Congestion']].copy()
    
    'Creating date/time features using datetime column Date as index'
    data_final = create_features(data_final)
     
    'Creating lag features of 1 day, 5 days and 1 week before'
    data_final = lag_features(data_final,[1,5,7], 'Congestion')
    data_final.fillna(0, inplace= True)
    data_final.dropna(inplace=True)
    
    'Transforming date/time features into two dimensional features'
    data_final= cyclical_features(data_final)
    
    'Removing outliers'
    data_final['Congestion']= np.where(data_final['Congestion'] < 0.06, data_final['Congestion'].mean(), data_final['Congestion'])
    
    'Array containing the names of all features available'
    all_features = data_final.columns.values.tolist()
    all_features.remove('Congestion')
    all_features= np.array(all_features) 
    
    X = data_final.values
    Y = X[:,0] 
    X = X[:,[x for x in range(1,len(all_features)+1)]]
    
    'Feature selection for the model'
    parameters_RF= {'bootstrap': True,
                    'min_samples_leaf': 3,
                    'n_estimators': 200, 
                    'min_samples_split': 7,
                    'max_depth': 30,
                    'max_leaf_nodes': None,
                    'random_state': 18}
    
    reg_RF= RandomForestRegressor(**parameters_RF)
    reg_RF.fit(X, Y)
    importance= pd.DataFrame(data= {'Feature': all_features, 'Score': reg_RF.feature_importances_})
    importance= importance.sort_values(by=['Score'], ascending= False)
    importance.set_index('Feature', inplace=True)
    
    'Defining the number of features to use in the models'
    num_features = 15  #Optimal number of features is 15
    
    'Defining training and test periods'
    data_train = data_final.loc[: start_forecast - timedelta(minutes= 15)]
    data_test = data_final.loc[start_forecast : start_forecast + relativedelta(months= 1) - timedelta(minutes= 15)]
    
    'Plot train-test '
    fig,ax = plt.subplots()
    coloring = data_final['Congestion'].max()
    plt.plot(data_train.index, data_train['Congestion'], color= "darkcyan", alpha= 0.75)
    plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
    plt.plot(data_test.index, data_test['Congestion'], color = "dodgerblue", alpha= 0.60)
    plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("PT capacity", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(['Train','Test'], frameon= False, loc= 'upper center', ncol= 2)
    plt.title(season+ " Train - Test split for Congestion Service", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Features used'
    USE_COLUMNS = importance[:num_features].index.values
    
    FORECAST_COLUMN = ['Congestion']
    
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
    plt.xlabel("Real PT capacity", alpha= 0.75, weight= "bold")
    plt.ylabel("Predicted PT capacity", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold",fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.title(season + " correlation real vs predictions for Congestion Service", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    #Real vs predictions in the same plot
    fig,ax = plt.subplots()
    ax.plot(df_RF.Real, label= "Real")
    ax.plot(df_RF.Prediction, label= "Predicted", ls= '--')
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("PT capacity", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold",fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(frameon= False, loc= 'best')
    plt.title(season + " real vs predicted for Congestion Service", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Errors'
    MAE_RF = metrics.mean_absolute_error(df_RF.Real, df_RF.Prediction)
    RMSE_RF = np.sqrt(metrics.mean_squared_error(df_RF.Real, df_RF.Prediction))
    normRMSE_RF = 100 * RMSE_RF / ytest['Congestion'].max()
    R2_RF = metrics.r2_score(df_RF.Real, df_RF.Prediction)
    
    print('RF - Mean Absolute Error (MAE):', round(MAE_RF,2))
    print('RF - Root Mean Square Error (RMSE):',  round(RMSE_RF,2))
    print('RF - Normalized RMSE (%):', round(normRMSE_RF,2))
    #print('RF - R square (%):', round(R2_RF,2))
    
###############################################################################################################################
    'Consumption Congestion Service Activation and ps cong sell (Regression)'
    print('#################################################################')
    print('Forecast variable: ', variables[0])
###############################################################################################################################

    df_RF[variables[0]] = np.where((df_RF.Prediction > 0.3), 1, 0)
    df_RF['Consumption Congestion'] = raw_data['Consumption Congestion Service Activation']
    
    'Accuracy'
    accuracy = accuracy_score(df_RF['Consumption Congestion'],  df_RF[variables[0]])
    print(f'Accuracy for {variables[0]}: {accuracy:.2f}')
    
    'Confusion matrix'
    cnf_matrix = confusion_matrix(df_RF['Consumption Congestion'],  df_RF[variables[0]], labels=[1,0])
    np.set_printoptions(precision=2)
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes= ['Activation= 1','Activation= 0'], normalize= False, title= season + ' confusion matrix for '+ variables[0])
    plt.show()
    
    print('#################################################################')
    print('Forecasting of ps cong sell')
    data_sell = data.loc[:, ['ps cong sell', variables[0]]].copy()
    
    'Creating date/time features using datetime column Date as index'
    data_sell = create_features(data_sell)
    
    'Transforming date/time features into two dimensional features'
    data_sell = cyclical_features(data_sell)
    
    'Defining training and test periods'
    data_train = data_sell.loc[: start_forecast - timedelta(minutes= 15)]
    data_test = data_sell.loc[start_forecast :]
    
    'Array containing the names of all features available'
    all_features = data_sell.columns.values.tolist()
    all_features.remove('ps cong sell')
    all_features= np.array(all_features) 
        
    X = data_sell.values
    Y = X[:, 0] 
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
    num_features = 3  #Optimal number of features is 3  
    
    'Plot train-test '
    fig,ax = plt.subplots()
    coloring = data_sell['ps cong sell'].max()
    plt.plot(data_train.index, data_train['ps cong sell'], color= "darkcyan", alpha= 0.75)
    plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
    plt.plot(data_test.index, data_test['ps cong sell'], color = "dodgerblue", alpha= 0.60)
    plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("ps cong sell", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(['Train','Test'], frameon= False, loc= 'upper center', ncol= 2)
    plt.title(season+ " Train - Test split for ps cong sell", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Features used'
    USE_COLUMNS = importance[:num_features].index.values
    print('The features used in the XGBOOST model are:', USE_COLUMNS)
    
    FORECAST_COLUMN = ['ps cong sell']
    
    'XGBOOST model'
    xtrain = data_train.loc[:, USE_COLUMNS]
    xtest = data_test.loc[:, USE_COLUMNS]
    ytrain = data_train.loc[:, FORECAST_COLUMN]
    ytest = data_test.loc[:, FORECAST_COLUMN]
    
    #Using the forecasted values of congestion cons to forecast ps cong sell
    if variables[0] in USE_COLUMNS:
        xtest[variables[0]] = df_RF[variables[0]]
        
    reg_XGBOOST.fit(xtrain, ytrain)
    
    'Predictions and Post-Processing'
    df_reg = pd.DataFrame(reg_XGBOOST.predict(xtest), columns= ['Prediction'], index= xtest.index)
    df_reg['Real'] = ytest
    df_reg[variables[0]] = df_RF[variables[0]]
    
    df_reg['Prediction']= np.where(df_reg[variables[0]] == 0, 1 , df_reg['Prediction'])
    
    'Real vs predictions plot'
    fig,ax = plt.subplots()
    ax.plot(df_reg.Real, label= "Real")
    ax.plot(df_reg.Prediction, label= "Predicted", ls= '--')
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("ps cong sell", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold",fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(frameon= False, loc= 'best')
    plt.title(season + " real vs predicted for ps cong sell", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Errors'
    MAE_XGBOOST = metrics.mean_absolute_error(df_reg.Real, df_reg.Prediction)
    RMSE_XGBOOST = np.sqrt(metrics.mean_squared_error(df_reg.Real, df_reg.Prediction))
    normRMSE_XGBOOST = 100 * RMSE_XGBOOST / ytest['ps cong sell'].max()
    R2_XGBOOST = metrics.r2_score(df_reg.Real, df_reg.Prediction)
    
    print('XGBOOST- Mean Absolute Error (MAE):', round(MAE_XGBOOST,2))
    print('XGBOOST - Root Mean Square Error (RMSE):',  round(RMSE_XGBOOST,2))
    print('XGBOOST - Normalized RMSE (%):', round(normRMSE_XGBOOST,2))
    #print('XGBOOST - R square (%):', round(R2_XGBOOST,2))
    
###############################################################################################################################
    'Generation Congestion Service Activation and ps cong buy (Regression)'
    print('#################################################################')
    print('Forecast variable: ', variables[1])
###############################################################################################################################

    df_RF[variables[1]] = np.where((df_RF.index.hour > 12) & (df_RF.index.hour < 20) & (df_RF.Prediction < 0.175), 1, 0)
    df_RF['Generation Congestion'] = raw_data['Generation Congestion Service Activation']
    
    'Accuracy'
    accuracy = accuracy_score(df_RF['Generation Congestion'],  df_RF[variables[1]])
    print(f'Accuracy for {variables[1]}: {accuracy:.2f}')
    
    'Confusion matrix'
    cnf_matrix = confusion_matrix(df_RF['Generation Congestion'],  df_RF[variables[1]], labels=[1,0])
    np.set_printoptions(precision=2)
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes= ['Activation= 1','Activation= 0'], normalize= False, title= season + ' confusion matrix for '+ variables[1])
    plt.show()
    
    print('#################################################################')
    print('Forecasting of ps cong buy')
    data_buy = data.loc[:, ['ps cong buy', variables[1]]].copy()
    
    'Creating date/time features using datetime column Date as index'
    data_buy = create_features(data_buy)
    
    'Transforming date/time features into two dimensional features'
    data_buy = cyclical_features(data_buy)
    
    'Defining training and test periods'
    data_train = data_buy.loc[: start_forecast - timedelta(minutes= 15)]
    data_test = data_buy.loc[start_forecast :]
    
    'Array containing the names of all features available'
    all_features = data_buy.columns.values.tolist()
    all_features.remove('ps cong buy')
    all_features= np.array(all_features) 
        
    X = data_buy.values
    Y = X[:, 0] 
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
    
    reg_XGBOOST2 = xgb.XGBRegressor(**parameters_XGBOOST)
    reg_XGBOOST2.fit(X, Y)
    importance = pd.DataFrame(data= {'Feature': all_features, 'Score': reg_XGBOOST2.feature_importances_})
    importance = importance.sort_values(by= ['Score'], ascending= False)
    importance.set_index('Feature', inplace= True)
    
    'Defining the number of features to use in the models'
    num_features = 3   #Optimal number of features is 3  
    
    'Plot train-test '
    fig,ax = plt.subplots()
    coloring = data_buy['ps cong buy'].max()
    plt.plot(data_train.index, data_train['ps cong buy'], color= "darkcyan", alpha= 0.75)
    plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
    plt.plot(data_test.index, data_test['ps cong buy'], color = "dodgerblue", alpha= 0.60)
    plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("ps cong buy", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(['Train','Test'], frameon= False, loc= 'upper center', ncol= 2)
    plt.title(season+ " Train - Test split for ps cong buy", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Features used'
    USE_COLUMNS = importance[:num_features].index.values
    print('The features used in the XGBOOST model are:', USE_COLUMNS)
    
    FORECAST_COLUMN = ['ps cong buy']
    
    'XGBOOST model'
    xtrain = data_train.loc[:, USE_COLUMNS]
    xtest = data_test.loc[:, USE_COLUMNS]
    ytrain = data_train.loc[:, FORECAST_COLUMN]
    ytest = data_test.loc[:, FORECAST_COLUMN]
    
    #Using the forecasted values of congestion gen to forecast ps cong buy
    if variables[1] in USE_COLUMNS:
        xtest[variables[1]] = df_RF[variables[1]]
        
    reg_XGBOOST2.fit(xtrain, ytrain)
    
    'Predictions and Post-Processing'
    df_reg2 = pd.DataFrame(reg_XGBOOST2.predict(xtest), columns= ['Prediction'], index= xtest.index)
    df_reg2['Real'] = ytest
    df_reg2[variables[1]] = df_RF[variables[1]]
    
    df_reg2['Prediction']= np.where(df_reg2[variables[1]] == 0, 1 , df_reg2['Prediction'])
    
    'Real vs predictions plot'
    fig,ax = plt.subplots()
    ax.plot(df_reg2.Real, label= "Real")
    ax.plot(df_reg2.Prediction, label= "Predicted", ls= '--')
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("ps cong buy", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold",fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(frameon= False, loc= 'best')
    plt.title(season + " real vs predicted for ps cong buy", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Errors'
    MAE_XGBOOST = metrics.mean_absolute_error(df_reg2.Real, df_reg2.Prediction)
    RMSE_XGBOOST = np.sqrt(metrics.mean_squared_error(df_reg2.Real, df_reg2.Prediction))
    normRMSE_XGBOOST = 100 * RMSE_XGBOOST / ytest['ps cong buy'].max()
    R2_XGBOOST = metrics.r2_score(df_reg2.Real, df_reg2.Prediction)
    
    print('XGBOOST- Mean Absolute Error (MAE):', round(MAE_XGBOOST,2))
    print('XGBOOST - Root Mean Square Error (RMSE):',  round(RMSE_XGBOOST,2))
    print('XGBOOST - Normalized RMSE (%):', round(normRMSE_XGBOOST,2))
    #print('XGBOOST - R square (%):', round(R2_XGBOOST,2))
    
    ###############################################################################################################################
    'Forecast results'
    ###############################################################################################################################
    
    predictions_congestion = df_reg.loc[:, [variables[0], 'Prediction']]
    predictions_congestion.rename(columns= {'Prediction' : 'ps cong sell'}, inplace= True)
    predictions_congestion = predictions_congestion.join(df_reg2.loc[:, [variables[1], 'Prediction']])
    predictions_congestion.rename(columns= {'Prediction' : 'ps cong buy'}, inplace= True)
    
    return predictions_congestion
    