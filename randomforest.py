from matplotlib import pylab as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np
import datetime as dt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import math

plt.rcParams['figure.figsize'] = (15.0, 8.0)

# Reading Dataset
df = pd.read_csv('Dataset1.csv', parse_dates=['Time'])
df = df.set_index('Time')
df1 = df[['PVPower', 'Outputactivepower', 'Outputvoltage', 'tempC', 'humidity',
          'pressure', 'sunHour', 'solar', 'uvIndex', 'cloudcover', 'FeelsLikeC']]

# Feature Engineering: Lagged values
for obs in range(1, 6):
    df1["T_" + str(obs)] = df1.PVPower.shift(obs)

df1.fillna(0.00, inplace=True)

clean_data = df1[['T_1','T_2', 'T_3', 'T_4', 'T_5', 'PVPower']]

# Splitting Data
training_data = df1[df1.index < pd.to_datetime("07/31/2017")]
val_data = df1[(df1.index >= pd.to_datetime("07/31/2017")) & 
               (df1.index < pd.to_datetime("08/06/2017"))]
test_data = df1[df1.index >= pd.to_datetime("08/06/2017")]

clean_train = training_data[['tempC', 'humidity','pressure', 'solar','cloudcover', 'T_1',
                             'T_2', 'T_3', 'T_4', 'T_5','PVPower']]
clean_test = test_data[['tempC', 'humidity','pressure', 'solar', 'cloudcover', 'T_1',
                        'T_2', 'T_3', 'T_4', 'T_5', 'PVPower']]
clean_val = val_data[['tempC', 'humidity', 'pressure', 'solar', 'cloudcover', 'T_1',
                      'T_2', 'T_3', 'T_4', 'T_5','PVPower']]

x_train, y_train = clean_train.drop(["PVPower"], axis=1), clean_train.PVPower
x_test, y_test = clean_test.drop(["PVPower"], axis=1), clean_test.PVPower
x_val, y_val = clean_val.drop(["PVPower"], axis=1), clean_val.PVPower

# Scaling
scaler = StandardScaler()
rfr = RandomForestRegressor(random_state=2017, verbose=2, n_jobs=5)

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
x_valid_scaled = scaler.fit_transform(x_val)

# Fitting Model
rfr.fit(x_train_scaled, y_train)

# Scoring
print("Val Score:", rfr.score(x_valid_scaled, y_val)) # Note: rfr.score takes (X, y)
print("Test Score:", rfr.score(x_test_scaled, y_test))

test_data["RF_PREDICTED"] = rfr.predict(x_test_scaled)
val_data["RF_PREDICTED"] = rfr.predict(x_valid_scaled)

# Evaluation
print("Mean squared error: ", mean_squared_error(val_data.PVPower, val_data.RF_PREDICTED))
rms = math.sqrt(mean_squared_error(val_data.PVPower, val_data.RF_PREDICTED))
print("Root mean squared error: ", rms)

print('R-squared score: ', r2_score(val_data.PVPower, val_data.RF_PREDICTED))
print('Mean absolute error: ', mean_absolute_error(val_data.PVPower, val_data.RF_PREDICTED))
print('Median absolute error: ', median_absolute_error(val_data.PVPower, val_data.RF_PREDICTED))
print('Correlation:', np.correlate(val_data.PVPower, val_data.RF_PREDICTED))