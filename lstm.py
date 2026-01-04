from matplotlib import pylab as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np
import datetime as dt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import math

plt.rcParams['figure.figsize'] = (15.0, 8.0)

# Reading Dataset
df = pd.read_csv('Dataset1.csv', parse_dates=['Time'])
df = df.set_index('Time')
df1 = df[['PVPower', 'Outputactivepower', 'Outputvoltage', 'tempC', 'humidity', 'pressure',
          'sunHour', 'solar', 'uvIndex','cloudcover', 'FeelsLikeC']]

for obs in range(1, 6):
    df1["T_" + str(obs)] = df1.PVPower.shift(obs)

df1.fillna(0.00, inplace=True)

training_data = df1[df1.index < pd.to_datetime("07/31/2017")]
val_data = df1[(df1.index >= pd.to_datetime("07/31/2017")) & 
               (df1.index < pd.to_datetime("08/06/2017"))]
test_data = df1[df1.index >= pd.to_datetime("08/06/2017")]

clean_train = training_data[['tempC', 'humidity', 'pressure', 'solar','cloudcover', 'Outputactivepower',
                             'T_1','T_2', 'T_3', 'T_4', 'T_5','PVPower']]
clean_test = test_data[['tempC', 'humidity', 'pressure', 'solar', 'cloudcover', 'Outputactivepower',
                        'T_1','T_2', 'T_3', 'T_4', 'T_5','PVPower']]
clean_val = val_data[['tempC', 'humidity', 'pressure', 'solar', 'cloudcover', 'Outputactivepower',
                      'T_1','T_2', 'T_3', 'T_4', 'T_5','PVPower']]

x_train, y_train = clean_train.drop(["PVPower"], axis=1), clean_train.PVPower
x_test, y_test = clean_test.drop(["PVPower"], axis=1), clean_test.PVPower
x_val, y_val = clean_val.drop(["PVPower"], axis=1), clean_val.PVPower

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
x_valid_scaled = scaler.fit_transform(x_val)

# Reshape for LSTM [samples, timesteps, features]
x_t_reshaped = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
x_val_reshaped = x_valid_scaled.reshape((x_valid_scaled.shape[0], 1, x_valid_scaled.shape[1]))

# Design Network
model = Sequential()
model.add(LSTM(25, input_shape=(x_t_reshaped.shape[1], x_t_reshaped.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# Fit Model
history = model.fit(x_t_reshaped, y_train, epochs=500, batch_size=250,
                    validation_data=(x_val_reshaped, y_val), verbose=2, shuffle=False)

# Plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make a prediction on test
x_te_reshaped = x_test_scaled.reshape((x_test_scaled.shape[0], 1, x_test_scaled.shape[1]))
yhat = model.predict(x_te_reshaped)

# Invert scaling for forecast
x_te_reshaped = x_te_reshaped.reshape((x_te_reshaped.shape[0], x_te_reshaped.shape[2]))
inv_yhat = np.concatenate((yhat, x_te_reshaped[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# Invert scaling for actual
y_test = y_test.values.reshape((len(y_test), 1))
inv_y = np.concatenate((y_test, x_te_reshaped[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

test_data["DL_PRED"] = inv_yhat
test_data["DL_Actual"] = inv_y

# Validation Prediction
yhat_val = model.predict(x_val_reshaped)
x_val_reshaped = x_val_reshaped.reshape((x_val_reshaped.shape[0], x_val_reshaped.shape[2]))

# Invert scaling for forecast
inv_yhatval = np.concatenate((yhat_val, x_val_reshaped[:, 1:]), axis=1)
inv_yhatval = scaler.inverse_transform(inv_yhatval)
inv_yhatval = inv_yhatval[:, 0]

# Invert scaling for actual
y_val = y_val.values.reshape((len(y_val), 1))
inv_yval = np.concatenate((y_val, x_val_reshaped[:, 1:]), axis=1)
inv_yval = scaler.inverse_transform(inv_yval)
inv_yval = inv_yval[:, 0]

val_data["DL_PRED"] = inv_yhatval
val_data["DL_Actual"] = inv_yval

# Metrics
print("Mean squared error: ", mean_squared_error(val_data.PVPower, val_data.DL_PRED))
rms = math.sqrt(mean_squared_error(val_data.PVPower, val_data.DL_PRED))
print("Root mean squared error: ", rms)

print('R-squared score: ', r2_score(val_data.PVPower, val_data.DL_PRED))
print('Mean absolute error: ', mean_absolute_error(val_data.PVPower, val_data.DL_PRED))
print('Median absolute error: ', median_absolute_error(val_data.PVPower, val_data.DL_PRED))
print('Correlation:', np.correlate(val_data.PVPower, val_data.DL_PRED))