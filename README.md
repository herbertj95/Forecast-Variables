# Forecast-Variables

Python code to do seasonal one-month forecast for different variables.

**Main.py**

Principal script, that calls the auxiliary forecast scripts for specific variables. 
The inputs are the following variables and the season (Winter, Spring, Summer, Fall) of the forecast.
- var1 = 'CP1 ID' 
- var2 = 'CP1 Plugged'
- var3 = 'CP1 req (kWh)'
- var4 = 'CP2 ID'
- var5 = 'CP2 Plugged'
- var6 = 'CP2 req (kWh)'
- var7 = 'PV gen #1 (kWh)'
- var8 = 'PV gen #2 (kWh)'
- var9 = 'PV gen #3 (kWh)'
- var10 = 'build cons #1 (kWh)'
- var11 = 'build cons #2 (kWh)'
- var12 = 'build cons #3 (kWh)'
- var13 = 'congestion mgmt cons (60% th)'
- var14 = 'congestion mgmt gen (17,5% th)'
- var15 = 'wind curtail (500 EVs)'

**Plug_and_Energy.py**

Auxiliary script that forecast var1,var2,var3 together and var4,var5,var6 together:
- Type of user (ID): Identifies the user of the EV that plugs into the charging point, it can be visitors (V), workers (W) or company fleet (F). Classification using Logistic Regression.
- Plugged: Defines if a EV is plugged (1) or not (0). Classification using XGBoost.
- Energy req: Energy demand (kWh) for current EV charging session. Regression using XGBoost. 

**PV_generation.py**

Auxiliary script that forecast var7, var8 and var9 individually:
- PV gen #1: PV generation corresponding to 15 kWp installed. Regression using XGBoost.
- PV gen #2: PV generation corresponding to 10 kWp installed. Regression using XGBoost.
- PV gen #3: PV generation corresponding to 20 kWp installed. Regression using XGBoost.

**Load.py**

Auxiliary script that forecast var10, var11 and var12 individually:
- build cons #1 (kWh): Standard building consumption corresponding to 100 MWh annually (excluding dispatchable loads). Regression using XGBoost.
- build cons #2 (kWh): Standard building consumption corresponding to 75 MWh annually (excluding dispatchable loads). Regression using XGBoost.
- build cons #3 (kWh): Standard building consumption corresponding to 125 MWh annually (excluding dispatchable loads). Regression using XGBoost.

**Congestion_service.py**

Auxiliary script that forecast var13 and var14 together:
- congestion mgmt cons (60% th): Indicates whether the user participates (1) or not (0) in network congestion management services assuming a 60% PT capacity threshold after n-1 PTs. Regression using Random Forest and Post-Processing using a threshold to determine the activation of the service (1) or not (0).
- congestion mgmt gen (17,5% th): Indicates whether the user participates (1) or not (0) in network congestion management services assuming a 17,5% PT capacity threshold. Regression using Random Forest and Post-Processing using a threshold to determine the activation of the service (1) or not (0).

**Wind_curtailment_service.py**
Auxiliary script that forecast var15:
- wind curtail (500 EVs): Indicates whether the user participates (1) or not (0) in the wind curtailment services and thus charges the EV in periods of wind surplus, assuming 500 participating EVs. Regression using Random Forest and Post-Processing using a threshold to determine the activation of the service (1) or not (0).
