import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import warnings
import itertools
import statsmodels.api as sm
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import math

# Reading Dataset
df = pd.read_csv('Dataset1.csv', parse_dates=['Time'])
df = df.set_index('Time')
df1 = df[['PVPower', 'Outputapparentpower', 'Outputactivepower', 'Outputvoltage',
          'tempC', 'humidity','pressure', 'sun Hour', 'solar', 'uvIndex', 'cloudcover', 
          'maxtempC', 'mintempC','FeelsLikeC', 'HeatIndexC', 'DewPointC']]

rcParams['figure.figsize'] = 11, 9
decomposition = sm.tsa.seasonal_decompose(df1.PVPower.values, model='additive', freq=24)
fig = decomposition.plot()
plt.show()

def test_stationarity(timeseries):
    # Determining rolling statistics
    rolmean = timeseries.rolling(window=12).mean() # Updated from pd.rolling_mean
    rolstd = timeseries.rolling(window=12).std()   # Updated from pd.rolling_std

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(df.PVPower)

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1])) 
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

warnings.filterwarnings("ignore") # specify to ignore warning messages
best = list()

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df1.PVPower,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            best.append(results.aic)
        except:
            continue

print('Best AIC Score:', min(best))

# Fitting the best model
mod = sm.tsa.statespace.SARIMAX(df1.PVPower,
                                order=(1, 0, 1),
                                seasonal_order=(1, 1, 1, 24),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# Prediction
pred = results.get_prediction(start=pd.to_datetime('2017-07-11'), dynamic=False)
pred_ci = pred.conf_int()

ax = df1.PVPower['2017':].plot(label='Actual')
pred.predicted_mean.plot(ax=ax, label='Predicted', alpha=.7)

plt.title('S-ARIMA')
ax.set_xlabel('Time')
ax.set_ylabel('PVPower (KWh)')
plt.legend()
plt.show()

y_forecasted = pred.predicted_mean
y_truth = df1.PVPower['2017-07-11':]
diff = y_forecasted - y_truth

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=24)
pred_ci = pred_uc.conf_int()

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.1)

ax.set_xlabel('DateTime')
ax.set_ylabel('PVPower(KWh)')
plt.title('S-ARIMA')
plt.legend()
plt.show()

print("Mean squared error: ", mean_squared_error(y_truth, y_forecasted))
rms = math.sqrt(mean_squared_error(y_truth, y_forecasted))
print("Root mean squared error: ", rms)

print('R-squared score: ', r2_score(y_truth, y_forecasted))
print('Mean absolute error: ', mean_absolute_error(y_truth, y_forecasted))
print('Median absolute error: ', median_absolute_error(y_truth, y_forecasted))
print('Correlation:', np.correlate(y_truth, y_forecasted))