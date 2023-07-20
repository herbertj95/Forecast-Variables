# -*- coding: utf-8 -*-
"""
Created on July 14 of 2023
Forecast Plug and Energy for EV4EU
@author: Herbert Amezquita
"""
###############################################################################################################################
'Libraries'
###############################################################################################################################
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report, accuracy_score
from sklearn import preprocessing
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
    if month_number in [1,2,3]:
        return 1
    elif month_number in [4,5,6]:
        return 2
    elif month_number in [7,8,9]:
        return 3
    elif month_number in [10,11,12]:
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
                                         'Dayofweek','Hour', 'Minute'])
    
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
'############################Forecasting of Plug (Classification) and Energy (Regression)######################################'
###############################################################################################################################

def forecast(data, variables, season, start_forecast):
    
    'Distributing energy consumed value equally during the period the EV is charging'
    data_final = data.loc[:, variables].copy()
    count = 0
    energy = None
    first_zero = True
    
    for i, value in enumerate(data_final[variables[0]]):
        # Cheking if the 'Plugged' is 1, if it is, count the number of ones and distribute the 'Energy req' value among them
        if value == 1:
            count += 1
            if data_final[variables[1]][i] != 0:
                energy = data_final[variables[1]][i]
                replacement = energy / count
                data_final.loc[i-count+1:i+1, variables[1]] = replacement
                count = 0
                energy = None
            
    'Looking for the errors in the distribution of the energy'
    error = data_final[(data_final[variables[0]] == 0) & (data_final[variables[1]] != 0) | (data_final[variables[0]] == 1) & (data_final[variables[1]] == 0)]

    'Creating date/time features using datetime column Date as index'
    data_final = create_features(data_final) 
    
    'Barplot average energy consumption per hour'
    # mean_per_hour = data_final.groupby('Hour')[variables[1]].agg(["mean"])
    # fig, ax = plt.subplots()
    # ax.plot(mean_per_hour.index, mean_per_hour["mean"], color= 'darkcyan')
    # plt.xticks(range(1,25), alpha= 0.75, weight= "bold")
    # plt.yticks(alpha=0.75, weight= "bold")
    # plt.xlabel("Hour", alpha=0.75, weight= "bold")
    # plt.ylabel("Energy (kWh)", alpha= 0.75, weight= "bold")
    # plt.title("Average energy consumption per hour", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
    # plt.show()
    
    'Transforming date/time features into two dimensional features'
    data_final = cyclical_features(data_final)
    
    'Creating lag features of 1 day, 5 days and 1 week before'
    data_final = lag_features(data_final,[1,5,7], variables[1])
    data_final.fillna(0, inplace= True)
    data_final.dropna(inplace=True)
    
    'Defining training and test periods'
    data_train = data_final.loc[: start_forecast - timedelta(minutes= 15)]
    data_test = data_final.loc[start_forecast :]
    
    'Forecasting Plug (Classification)'
    print('Forecast variable: ', variables[0])
    
    'Plot train-test'
    fig,ax = plt.subplots()
    coloring = data_final[variables[0]].max()
    plt.plot(data_train.index, data_train[variables[0]], color= "darkcyan", alpha= 0.75)
    plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
    plt.plot(data_test.index, data_test[variables[0]], color = "dodgerblue", alpha= 0.60)
    plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("Plugged (1), Unplugged (0)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(['Train','Test'], frameon= False, loc= 'upper center', ncol= 2)
    plt.title(season+ " Train - Test split for "+ variables[0], alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    xtrain = data_train.iloc[:, (data_train.columns != variables[0]) & (data_train.columns != variables[1])]
    xtest = data_test.iloc[:, (data_train.columns != variables[0]) & (data_train.columns != variables[1])]

    ytrain = data_train.loc[:, [variables[0]]]
    ytest = data_test.loc[:, [variables[0]]]
    
    #XGBOOST
    cla_XGBOOST = xgb.XGBClassifier()

    cla_XGBOOST.fit(xtrain, ytrain)
    
    #Predictions
    df_cla= pd.DataFrame(cla_XGBOOST.predict(xtest), columns=['Prediction'], index= xtest.index)
    df_cla['Real']= ytest

    #Accuracy
    accuracy = accuracy_score(df_cla.Real, df_cla.Prediction)
    print(f'XGBOOST Classifier Accuracy for Plug: {accuracy:.2f}')
    
    #Confusion matrix
    cnf_matrix = confusion_matrix(df_cla.Real, df_cla.Prediction, labels=[1,0])
    np.set_printoptions(precision=2)
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes= ['Plugged= 1','Plugged= 0'], normalize= False, title= season + ' confusion matrix for '+ variables[0])
    plt.show()
    
    #Real vs predictions in the same plot
    fig,ax = plt.subplots()
    ax.plot(df_cla.Real, label= "Real")
    ax.plot(df_cla.Prediction, label= "Predicted", ls= '--')
    plt.xlabel("Date", alpha= 0.75, weight="bold")
    plt.ylabel('Plugged (1), Unplugged (0)', alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75, weight= "bold",fontsize= 11)
    plt.yticks(alpha= 0.75, weight= "bold", fontsize= 11)
    plt.legend(frameon= False, loc= 'best')
    plt.title(season+ " correlation real vs predicted for "+ variables[0], alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Forecasting Energy (Regression)'
    print('Forecast variable: ', variables[1])
    
    'Plot train-test'
    fig,ax = plt.subplots()
    coloring = data_final[variables[1]].max()
    plt.plot(data_train.index, data_train[variables[1]], color= "darkcyan", alpha= 0.75)
    plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
    plt.plot(data_test.index, data_test[variables[1]], color = "dodgerblue", alpha= 0.60)
    plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("Energy (kWh)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
    plt.legend(['Train','Test'], frameon= False, loc= 'upper center', ncol= 2)
    plt.title(season+ " Train - Test split for " + variables[1], alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    'Array containing the names of all features available'
    all_features = data_final.columns.values.tolist()
    all_features.remove(variables[1])
    all_features= np.array(all_features) 
    
    'Moving the energy column to the beginning of the dataframe'
    first_column = data_final.pop(variables[1])
    data_final.insert(0, variables[1], first_column)
        
    X = data_final.values
    Y = X[:, 0] 
    X = X[:,[x for x in range(1,len(all_features)+1)]]
    
    'Feature selection for the model'
    reg_XGBOOST = xgb.XGBRegressor()
    reg_XGBOOST.fit(X, Y)
    importance = pd.DataFrame(data= {'Feature': all_features, 'Score': reg_XGBOOST.feature_importances_})
    importance = importance.sort_values(by= ['Score'], ascending= False)
    importance.set_index('Feature', inplace= True)
    
    'Defining the number of features to use in the models'
    num_features = 15   #Optimal number of features is 15  

    'Features used'
    USE_COLUMNS = importance[:num_features].index.values
    print('The features used in the XGBOOST model are:', USE_COLUMNS)
    
    FORECAST_COLUMN = [variables[1]]
    
    'XGBOOST model'
    xtrain = data_train.loc[:, USE_COLUMNS]
    xtest = data_test.loc[:, USE_COLUMNS]
    ytrain = data_train.loc[:, FORECAST_COLUMN]
    ytest = data_test.loc[:, FORECAST_COLUMN]
    
    #Using the forecasted values of Plug to forecast the Energy
    if variables[0] in USE_COLUMNS:
        xtest[variables[0]] = df_cla.Prediction
        
    reg_XGBOOST.fit(xtrain, ytrain)

    #Predictions and Pos-Processing
    df_reg = pd.DataFrame(reg_XGBOOST.predict(xtest), columns= ['Prediction'], index= xtest.index)
    df_reg['Real'] = ytest
    df_reg[variables[0]] = df_cla.Prediction
    
    df_reg['Prediction'] = np.where((df_reg['Prediction'] < 0) | (df_reg[variables[0]] == 0) & (df_reg.Prediction != 0) , 0 , df_reg['Prediction'])
    df_reg[variables[0]] =  np.where((df_reg[variables[0]] == 1) & (df_reg.Prediction == 0), 0, df_reg[variables[0]])
    
    #Regression Plot
    sns.scatterplot(data= df_reg, x= 'Real', y= 'Prediction')
    plt.plot(ytest, ytest, color = "dodgerblue", linewidth= 2) 
    plt.xlabel("Real energy (kWh)", alpha= 0.75, weight= "bold")
    plt.ylabel("Predicted energy (kWh)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75, weight= "bold", fontsize= 11)
    plt.yticks(alpha= 0.75, weight= "bold", fontsize= 11)
    plt.title(season+" correlation real vs predictions for "+ variables[1], alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    #Real vs predictions in the same plot
    fig,ax = plt.subplots()
    ax.plot(df_reg.Real, label= "Real")
    ax.plot(df_reg.Prediction, label= "Predicted", ls= '--')
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("Energy (kWh)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75, weight= "bold",fontsize= 11)
    plt.yticks(alpha= 0.75, weight= "bold", fontsize= 11)
    plt.legend(frameon= False, loc= 'best')
    plt.title(season+ " real vs predicted for "+ variables[1], alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()
    
    #Errors
    MAE_XGBOOST = metrics.mean_absolute_error(df_reg.Real, df_reg.Prediction)
    RMSE_XGBOOST = np.sqrt(metrics.mean_squared_error(df_reg.Real, df_reg.Prediction))
    normRMSE_XGBOOST = 100 * RMSE_XGBOOST / ytest[variables[1]].max()
    R2_XGBOOST = metrics.r2_score(df_reg.Real, df_reg.Prediction)
    
    print('XGBOOST- Mean Absolute Error (MAE):', round(MAE_XGBOOST,2))
    print('XGBOOST - Root Mean Square Error (RMSE):',  round(RMSE_XGBOOST,2))
    print('XGBOOST - Normalized RMSE (%):', round(normRMSE_XGBOOST,2))
    print('XGBOOST - R square (%):', round(R2_XGBOOST,2))
    
    'Forecast result'
    predictions_plug_energy = df_reg[[variables[0]]]
    predictions_plug_energy[variables[1]] = df_reg.Prediction
    
    #Post-processing
    count = 0
    energy_cum = 0
    
    for i, value in enumerate(predictions_plug_energy[variables[0]]):
        # Cheking if the 'Plugged' is 1, if it is, count the number of ones and sum the energy
        if value == 1:
            first_zero = False
            count += 1
            energy_cum += predictions_plug_energy[variables[1]][i]
        # Checking if the 'Plugged' is 0, if it is, assign the energy_cum to the last row
        if value == 0 and first_zero is False:
            first_zero = True
            predictions_plug_energy.loc[i-count:i-1, variables[1]] = 0
            predictions_plug_energy[variables[1]][i-1] = energy_cum
            count = 0
            energy_cum = 0
        # For the end of the dataset   
        if i == len(predictions_plug_energy)-1 and value == 1:
            first_zero = True
            predictions_plug_energy.loc[i-count:i, variables[1]] = 0
            predictions_plug_energy[variables[1]][i] = energy_cum
            count = 0
            energy_cum = 0
            
    return predictions_plug_energy 
