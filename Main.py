# -*- coding: utf-8 -*-
"""
Created on July 12 of 2023
Forecast Variables Simulator for EV4EU
@author: Herbert Amezquita
"""
###############################################################################################################################
'Libraries'
###############################################################################################################################
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings

import Plug_and_Energy
import PV_generation
import Load
import Congestion_service
import Wind_curtailment_service

###############################################################################################################################
'Plot Parameters'
###############################################################################################################################
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20})
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

###############################################################################################################################
'Start of the pipeline'
print(datetime.now())
print('##############################Forecast Variables Simulator-EV4EU Project###############################')

###############################################################################################################################
'Inputs'
print('Inputs')
###############################################################################################################################

'Define the variable(s) to forecast, in a list'
var1 = 'CP1 ID'
var2 = 'CP1 Plugged'
var3 = 'CP1 req (kWh)'
var4 = 'CP2 ID'
var5 = 'CP2 Plugged'
var6 = 'CP2 req (kWh)'
var7 = 'PV gen #1 (kWh)'
var8 = 'PV gen #2 (kWh)'
var9 = 'PV gen #3 (kWh)'
var10 = 'build cons #1 (kWh)'
var11 = 'build cons #2 (kWh)'
var12 = 'build cons #3 (kWh)'
var13 = 'congestion mgmt cons (60% th)'
var14 = 'congestion mgmt gen (17,5% th)'
var15 = 'wind curtail (500 EVs)'

forecast_var = [var1, var2, var3, var4, var5, var6, var7, var8, var9, var10,
                var11, var12, var13, var14, var15]

# forecast_var = [var1,var2,var3]

print('Forecasting variable(s):', forecast_var)

'Define the starting dates of the monthly forecast (the end dates will be one month after)'
winter_start = datetime.strptime('2019-02-20', '%Y-%m-%d')
spring_start = datetime.strptime('2019-05-21', '%Y-%m-%d')
summer_start = datetime.strptime('2019-08-23', '%Y-%m-%d')
fall_start = datetime.strptime('2019-11-21', '%Y-%m-%d')

'Define the forecast season(s) in a list (Winter, Spring, Summer or Fall)'
forecast_period = ['Winter', 'Spring', 'Summer', 'Fall']

# forecast_period = ['Winter']

print('Forecasting season(s): ', forecast_period)

###############################################################################################################################
'Data and dataframes'
###############################################################################################################################

'Reading the raw data'
data = pd.read_csv('./Datasets/Data2.csv', parse_dates= ['date (dd/mm/yy hh:mm)'])
data.rename(columns = {'date (dd/mm/yy hh:mm)' : 'Date'}, inplace= True)
data.set_index('Date', inplace= True)

print('Missing values in the dataset: ', data.isna().sum().sum())

for season in forecast_period:
    
    print(f'#########################################Season: {season}###############################################')
    
    'Filtering the data based on the season to forecast'
    if season == 'Winter':
        data_filtered = data.loc[: winter_start + relativedelta(months= 1) - timedelta(minutes= 15)]
        start_forecast = winter_start
        
    elif season == 'Spring':
        data_filtered = data.loc[: spring_start + relativedelta(months= 1) - timedelta(minutes= 15)]
        start_forecast = spring_start
        
    elif season == 'Summer':
        data_filtered = data.loc[: summer_start + relativedelta(months= 1) - timedelta(minutes= 15)]
        start_forecast = summer_start
        
    elif season == 'Fall':
        data_filtered = data.loc[: fall_start + relativedelta(months= 1) - timedelta(minutes= 15)]
        start_forecast = fall_start
    
    'Creating the dataframe(s) that will contain the predictions and the errors'
    if season == 'Winter':
        forecast_winter= pd.DataFrame(index= pd.date_range(start=winter_start, end=winter_start + relativedelta(months= 1) - timedelta(minutes= 15), freq='15min')) 
        forecast_winter.index.name='Date'
    
    elif season == 'Spring':
        forecast_spring= pd.DataFrame(index= pd.date_range(start=spring_start, end=spring_start + relativedelta(months= 1) - timedelta(minutes= 15), freq='15min')) 
        forecast_spring.index.name='Date'
        
    elif season == 'Summer':
        forecast_summer= pd.DataFrame(index= pd.date_range(start=summer_start, end=summer_start + relativedelta(months= 1) - timedelta(minutes= 15), freq='15min')) 
        forecast_summer.index.name='Date'
        
    elif season == 'Fall':
        forecast_fall= pd.DataFrame(index= pd.date_range(start=fall_start, end=fall_start + relativedelta(months= 1) - timedelta(minutes= 15), freq='15min')) 
        forecast_fall.index.name='Date'
        
    print(f'Training period: {data_filtered.index[0].date()} to {(start_forecast - timedelta(minutes= 15)).date()}')
    print(f'Forecast period: {start_forecast.date()} to {data_filtered.index[-1].date()}')
        
###############################################################################################################################
    'Forecasting'
###############################################################################################################################
    
###############################################################################################################################
    print('##################################Plug and Energy#####################################')
###############################################################################################################################
    
    'Checking that plug and energy variables are included together in forecast_var'
    if var2 in forecast_var and var3 not in forecast_var or var2 not in forecast_var and var3 in forecast_var:
        print(f'To forecast plug and energy you need to include both var2[{var2}] and var3[{var3}] in Forecasting variable(s)')
        break
    
    if var4 in forecast_var and var5 not in forecast_var or var4 not in forecast_var and var5 in forecast_var:
        print(f'To forecast plug and energy you need to include both var4[{var4}] and var5[{var5}] in Forecasting variable(s)')
        break
    
    'Forecast of var1, var2 and var3: CP1 ID, CP1 Plugged and CP1 req (kWh)'
    if var1 and var2 and var3 in forecast_var:
        cp1_plug_energy = Plug_and_Energy.forecast(data_filtered, [var1, var2, var3], season, start_forecast)
        
        #Adding the predictions to the dataframes(s)
        if season == 'Winter':
            forecast_winter = forecast_winter.join(cp1_plug_energy)
            
        elif season == 'Spring':
            forecast_spring = forecast_spring.join(cp1_plug_energy)
            
        elif season == 'Summer':
            forecast_summer = forecast_summer.join(cp1_plug_energy)
            
        elif season == 'Fall':
            forecast_fall = forecast_fall.join(cp1_plug_energy)
        
    'Forecast of var4, var5 and var6: CP2 ID, CP2 Plugged and CP2 req (kWh)'
    if var4 and var5 and var6 in forecast_var:
        cp2_plug_energy = Plug_and_Energy.forecast(data_filtered, [var4, var5, var6], season, start_forecast)
        
        #Adding the predictions to the dataframes(s)
        if season == 'Winter':
            forecast_winter = forecast_winter.join(cp2_plug_energy)
            
        elif season == 'Spring':
            forecast_spring = forecast_spring.join(cp2_plug_energy)
            
        elif season == 'Summer':
            forecast_summer = forecast_summer.join(cp2_plug_energy)
            
        elif season == 'Fall':
            forecast_fall = forecast_fall.join(cp2_plug_energy)
            
###############################################################################################################################
    print('###################################PV Generation######################################')
###############################################################################################################################
    
    'Forecast of var7: PV gen #1 (kWh)'
    if var7 in forecast_var:
        pvgen1 = PV_generation.forecast(data_filtered, var7, season, start_forecast)
        
        #Adding the predictions to the dataframes(s)
        if season == 'Winter':
            forecast_winter = forecast_winter.join(pvgen1)
            
        elif season == 'Spring':
            forecast_spring = forecast_spring.join(pvgen1)
            
        elif season == 'Summer':
            forecast_summer = forecast_summer.join(pvgen1)
            
        elif season == 'Fall':
            forecast_fall = forecast_fall.join(pvgen1)
            
    'Forecast of var8: PV gen #2 (kWh)'
    if var8 in forecast_var:
        pvgen2 = PV_generation.forecast(data_filtered, var8, season, start_forecast)
        
        #Adding the predictions to the dataframes(s)
        if season == 'Winter':
            forecast_winter = forecast_winter.join(pvgen2)
            
        elif season == 'Spring':
            forecast_spring = forecast_spring.join(pvgen2)
            
        elif season == 'Summer':
            forecast_summer = forecast_summer.join(pvgen2)
            
        elif season == 'Fall':
            forecast_fall = forecast_fall.join(pvgen2)
            
    'Forecast of var9: PV gen #3 (kWh)'
    if var9 in forecast_var:
        pvgen3 = PV_generation.forecast(data_filtered, var9, season, start_forecast)
        
        #Adding the predictions to the dataframes(s)
        if season == 'Winter':
            forecast_winter = forecast_winter.join(pvgen3)
            
        elif season == 'Spring':
            forecast_spring = forecast_spring.join(pvgen3)
            
        elif season == 'Summer':
            forecast_summer = forecast_summer.join(pvgen3)
            
        elif season == 'Fall':
            forecast_fall = forecast_fall.join(pvgen3)
            
###############################################################################################################################
    print('###########################Load (Building Consumption)################################')
###############################################################################################################################
            
    'Forecast of var10: build cons #1 (kWh)'
    if var10 in forecast_var:
        load1 = Load.forecast(data_filtered, var10, season, start_forecast)
        
        #Adding the predictions to the dataframes(s)
        if season == 'Winter':
            forecast_winter = forecast_winter.join(load1)
            
        elif season == 'Spring':
            forecast_spring = forecast_spring.join(load1)
            
        elif season == 'Summer':
            forecast_summer = forecast_summer.join(load1)
            
        elif season == 'Fall':
            forecast_fall = forecast_fall.join(load1)
    
    'Forecast of var11: build cons #2 (kWh)'
    if var11 in forecast_var:
        load2 = Load.forecast(data_filtered, var11, season, start_forecast)
        
        #Adding the predictions to the dataframes(s)
        if season == 'Winter':
            forecast_winter = forecast_winter.join(load2)
            
        elif season == 'Spring':
            forecast_spring = forecast_spring.join(load2)
            
        elif season == 'Summer':
            forecast_summer = forecast_summer.join(load2)
            
        elif season == 'Fall':
            forecast_fall = forecast_fall.join(load2)
            
    'Forecast of var12: build cons #3 (kWh)'
    if var12 in forecast_var:
        load3 = Load.forecast(data_filtered, var12, season, start_forecast)
        
        #Adding the predictions to the dataframes(s)
        if season == 'Winter':
            forecast_winter = forecast_winter.join(load3)
            
        elif season == 'Spring':
            forecast_spring = forecast_spring.join(load3)
            
        elif season == 'Summer':
            forecast_summer = forecast_summer.join(load3)
            
        elif season == 'Fall':
            forecast_fall = forecast_fall.join(load3)

###############################################################################################################################
    print('###############################Congestion Service#####################################')
###############################################################################################################################
    
    'Checking that congestion variables are included together in forecast_var'
    if var13 in forecast_var and var14 not in forecast_var or var13 not in forecast_var and var14 in forecast_var:
        print(f'To forecast congestion service you need to include both var13[{var13}] and var14[{var14}] in Forecasting variable(s)')
        break
    
    'Forecast of var13 and var14: congestion mgmt cons (60% th) and congestion mgmt gen (17,5% th)'
    if var13 and var14 in forecast_var:
        congestion = Congestion_service.forecast(data_filtered, [var13, var14], season, start_forecast)
        
        #Adding the predictions to the dataframes(s)
        if season == 'Winter':
            forecast_winter = forecast_winter.join(congestion)
            
        elif season == 'Spring':
            forecast_spring = forecast_spring.join(congestion)
            
        elif season == 'Summer':
            forecast_summer = forecast_summer.join(congestion)
            
        elif season == 'Fall':
            forecast_fall = forecast_fall.join(congestion)

###############################################################################################################################
    print('#############################Wind Curtailment Service#################################')
###############################################################################################################################

    'Forecast of var15: wind curtail (500 EVs)'
    if var15 in forecast_var:
        wind_curtailment = Wind_curtailment_service.forecast(data_filtered, var15, season, start_forecast)
        
        #Adding the predictions to the dataframes(s)
        if season == 'Winter':
            forecast_winter = forecast_winter.join(wind_curtailment)
            
        elif season == 'Spring':
            forecast_spring = forecast_spring.join(wind_curtailment)
            
        elif season == 'Summer':
            forecast_summer = forecast_summer.join(wind_curtailment)
            
        elif season == 'Fall':
            forecast_fall = forecast_fall.join(wind_curtailment)
            
###############################################################################################################################
'Saving the results'
###############################################################################################################################

print('###################################Final CSV(s)#######################################')

'Saving the csv(s) with the predictions'   
if 'Winter' in forecast_period:
    forecast_winter.to_csv('./Forecast Results/Winter.csv', encoding='utf-8', index=True)
    print('csv with the winter predictions saved')
    
if 'Spring' in forecast_period:
    forecast_spring.to_csv('./Forecast Results/Spring.csv', encoding='utf-8', index=True)
    print('csv with the spring predictions saved')
    
if 'Summer' in forecast_period:
    forecast_summer.to_csv('./Forecast Results/Summer.csv', encoding='utf-8', index=True)
    print('csv with the summer predictions saved')
    
if 'Fall' in forecast_period:
    forecast_fall.to_csv('./Forecast Results/Fall.csv', encoding='utf-8', index=True)
    print('csv with the fall predictions saved')

###############################################################################################################################
'End of the pipeline'
print(datetime.now())
###############################################################################################################################