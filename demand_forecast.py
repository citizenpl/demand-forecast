import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import pearsonr,spearmanr,describe
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold as CV_kfold
from sklearn.metrics import mean_squared_error
from datetime import datetime
from matplotlib import pyplot
from math import sqrt
#input processing
import os

def cross_val_set(X,Y):
    cv = CV_kfold(n_splits=5, shuffle=False, random_state=None)
    X_train = [];
    Y_train = [];
    X_val = [];
    Y_val = []
    for train, validate in cv.split(X,Y):
        x_train = X[train]
        y_train = Y[train]
        x_val = X[validate]
        y_val = Y[validate]
        X_train.append(x_train)
        Y_train.append(y_train)
        X_val.append(x_val)
        Y_val.append(y_val)
    eval_set=[[X_train,Y_train],[X_val,Y_val]]
    return {'cv':cv,'eval_set':eval_set}

directory='C:/Users/plgeo/OneDrive/PC Desktop/demand forecast/'
filetosave0=directory+'Processing/Energy_signals_correlation.xlsx'
filetowrite0=pd.ExcelWriter(filetosave0, engine='xlsxwriter')##excel file to save processed data'
files=[x for x in os.listdir(directory) if x.endswith('.csv')]
Population={'Valencia':791413,'Madrid':3223334,'Bilbao':345821,' Barcelona':1620343,'Seville':684324}
warm_months=["04","05","06","07","08","09","10"]
cold_months=["11","12","01","02","03"]
fileweather=directory+files[1]
fileenergy=directory+files[0]
data_weather=pd.read_csv(fileweather)
data_energy=pd.read_csv(fileenergy)
data_energy.rename(columns={'time':'timestamp'},inplace='True')
output_dates=data_energy.loc[:,'timestamp'].values
output_dates=[output.split('+')[0] for index,output in enumerate(output_dates)]
data_energy.dropna(axis=1,how='all',inplace=True)## drop columns with completely missing values
data_energy.drop(labels=['generation fossil coal-derived gas','generation fossil oil shale','generation fossil peat','generation geothermal','generation marine','generation wind offshore'],axis=1,inplace=True)
data_energy.fillna(axis=0,method='ffill',inplace=True)## fill missing values with the last valid observation
energy_corr=data_energy.corr(method='pearson')
energy_corr.to_excel(filetowrite0, sheet_name='Sheet1', engine="xlsxwriter", startrow=1, startcol=0, header=True)
filetowrite0.close()
df_generation_thermal_distributed_RES=data_energy.iloc[:,1:12].sum(axis=1)
df_forecast_solar=pd.DataFrame(data_energy.loc[:,'forecast solar day ahead'],columns=['forecast solar day ahead'])
df_forecast_wind=pd.DataFrame(data_energy.loc[:,'forecast wind onshore day ahead'],columns=['forecast wind onshore day ahead'])
df_forecast_demand=data_energy.loc[:,'total load forecast']
df_actual_demand=data_energy.loc[:,'total load actual']
ind_outnan=list(np.where(df_actual_demand.isna())[0])## find indices of nan values
### df_actual_demand.drop(labels=ind_outnan,inplace=True) ## drop specific indices from the pandas series while maintaining the same pandas data series
###output_dates=[j for i,j in enumerate(output_dates) if i not in ind_outnan]
df_actual_demand=pd.DataFrame(list(df_actual_demand),index=output_dates,columns=['total load actual'])
ind_outnan=list(np.where(df_actual_demand.isna())[0])## find indices of nan values
list_of_cities=list(dict.fromkeys(list(data_weather['city_name'])))## returns a list of the cities involved
raw_of_cities=data_weather['city_name'].values
## CORRELATE DEMAND with weather_features
input_variable_names=list(data_weather.columns.values)
input_variable_names.remove('dt_iso')
input_variable_names.remove('city_name')
input_variable_names.remove('rain_3h')
input_variable_names.remove('snow_3h')
input_variable_names.remove('wind_deg')

Weather_variables_warm,Weather_variables_cold={},{}
Input_variables_cold,Input_variables_warm={},{}

for city in list_of_cities:
    index_city = [ind for ind, val in enumerate(raw_of_cities) if val == city]
    population_no=Population[city]## number of people living in each city
    temp = data_weather.loc[index_city, 'temp'].values
    temp_min = data_weather.loc[index_city, 'temp_min'].values
    temp_max = data_weather.loc[index_city, 'temp_max'].values
    pressure = data_weather.loc[index_city, 'pressure'].values
    humidity = data_weather.loc[index_city, 'humidity'].values
    wind_speed = data_weather.loc[index_city, 'wind_speed'].values
    precip = data_weather.loc[index_city, 'rain_1h'].values
    cloudiness=data_weather.loc[index_city,'clouds_all']
    input_dates = data_weather.loc[index_city, 'dt_iso'].values
    input_dates = [input.split('+')[0] for index, input in enumerate(input_dates)]
    input_indices = list(dict.fromkeys([ind for ind, val in enumerate(output_dates) if val in input_dates]))
    indices_warm_season = list(dict.fromkeys([ind for ind, val in enumerate(output_dates) if val in input_dates and val.split('-')[1] in warm_months]))
    indices_cold_season=list(dict.fromkeys([ind for ind, val in enumerate(output_dates) if val in input_dates and val.split('-')[1] in cold_months]))
    df_weather = pd.DataFrame(np.column_stack([temp_max, pressure, humidity, wind_speed, precip,cloudiness]),columns=input_variable_names[2:8])  ## create a new dataframe of weather features for each city
    df_weather_variables_help=df_weather.iloc[input_indices,]## keep rows with specific indices and maintain the dataframe
    df_weather_variables = df_weather_variables_help.copy()
    df_weather_variables_warm=df_weather.iloc[indices_warm_season,:]
    df_weather_variables_cold=df_weather.iloc[indices_cold_season,:]
    # DF_weather_variables = DF_weather_variables_help.copy()
    counter_corr_warm,counter_corr_cold=0,0
    for name in df_weather_variables.columns.values:
#        # df_weather_variables.loc[:,name]*=population_no/sum(Population.values())## multiply a column of pandas dataframe with the population ratio of each city
# #     #     ind_nan = list(np.where(df_weather_variables[name].isna()))  ## find indices of nan values
        input_warm=np.array(df_weather_variables_warm.loc[:,name].values)
        output_warm=np.array(df_actual_demand.iloc[indices_warm_season].values)/1000
        input_cold=np.array(df_weather_variables_cold.loc[:,name].values)
        output_cold=np.array(df_actual_demand.iloc[indices_cold_season].values)/1000
        #corr=spearmanr(input,output,alternative='two-sided')## spearmanr correlation with significance test for correlation.
        p_corr_warm=pearsonr(input_warm,output_warm)[1]
        p_corr_cold=pearsonr(input_cold,output_cold)[1]
        ## keep correlated variables only
        if p_corr_warm<0.001:
           counter_corr_warm+=1
        else:
            df_weather_variables_warm.drop([name],axis=1)
        if p_corr_cold<0.001:
           counter_corr_cold+=1
        else:
            df_weather_variables_cold.drop([name],axis=1)
    # ##shortlist number of cities based on correlations of the explanatory variables with demand.
    ## the explanatory variables are shortlisted based on their correlation too.
    if counter_corr_warm>len(df_weather_variables.columns.values)/2:
       Weather_variables_warm[city]=df_weather_variables_warm ## stores the transformed weather variables of each city with correlated variables , which will serve to produce a mean timeseries of each weather variable for the whole country by averaging
    if counter_corr_cold>len(df_weather_variables.columns.values)/2:
       Weather_variables_cold[city] = df_weather_variables_cold

from functools import reduce
DF_weather_variables_warm_help=reduce(lambda a,b: a.add(b,fill_value=0),Weather_variables_warm.values()) ## sum the elements of each dataframe representing each city to get a country-level set of weather features
DF_weather_variables_warm=DF_weather_variables_warm_help.apply(lambda x: x/len(Weather_variables_warm.values())) ## compute the mean of the dataframes elementwise
DF_weather_variables_cold_help=reduce(lambda a,b: a.add(b,fill_value=0),Weather_variables_cold.values()) ## sum the elements of each dataframe representing each city to get a country-level set of weather features
DF_weather_variables_cold=DF_weather_variables_cold_help.apply(lambda x: x/len(Weather_variables_cold.values())) ## compute the mean of the dataframes elementwise


column_names=list(DF_weather_variables_warm.columns.values)
column_names.append('forecast_solar')
column_names.append('forecast_wind')
DF_input_variables_warm=pd.concat([DF_weather_variables_warm,pd.concat([df_forecast_solar.iloc[indices_warm_season,:],df_forecast_wind.iloc[indices_warm_season,:]],axis=1,ignore_index=True)],axis=1,ignore_index=True)
DF_input_variables_warm.columns=column_names
DF_input_variables_cold=pd.concat([DF_weather_variables_cold,pd.concat([df_forecast_solar.iloc[indices_cold_season,:],df_forecast_wind.iloc[indices_cold_season,:]],axis=1,ignore_index=True)],axis=1,ignore_index=True)
DF_input_variables_cold.columns=column_names

column_names.append('total load actual')
DF_var_cold=pd.concat([DF_input_variables_cold.reset_index(drop=True),df_actual_demand.iloc[indices_cold_season].reset_index(drop=True)],axis=1)## this concatenation issues nan values unless we choose to reset_index in both dataframes
DF_var_cold.columns=column_names
DF_var_cold.index=df_actual_demand.iloc[indices_cold_season].index

DF_var_warm=pd.concat([DF_input_variables_warm.reset_index(drop=True),df_actual_demand.iloc[indices_warm_season].reset_index(drop=True)],axis=1)
DF_var_warm.columns=column_names
DF_var_warm.index=df_actual_demand.iloc[indices_warm_season].index
# for name in DF_var_cold.columns.values:
#     print(list(np.where(DF_var_cold[name].isna())))
# #     input=np.array(DF_weather_variables_cold.loc[:,name].values)
# #     output=np.array(df_actual_demand.iloc[indices_cold_season].values)/1000
# #     overall_correlation=pearsonr(input,output)[0]
# #

# dates_warm=[datetime.strptime(da,'%Y-%m-%d %H') for inde,da in enumerate(output_dates) if inde in indices_warm_season]
# dates_cold=[datetime.strptime(da,'%Y-%m-%d %H') for inde,da in enumerate(output_dates) if inde in indices_cold_season]
# #
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# convert series to supervised learning , returns the desired lag timesteps with n_in
def series_to_supervised(data, n_in=1, n_out=1,dropna=True):
 n_vars = 1 if type(data) is list else data.shape[1]
 df=pd.DataFrame(data)
 cols, names = list(), list()
 # input sequence (t-n, ... t-1)
 for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
 # forecast sequence (t, t+1, ... t+n)
 for i in range(0, n_out):
     cols.append(df.shift(-i))
 if i == 0:
    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
 else:
    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
 # put it all together
 agg = pd.concat(cols, axis=1)
 agg.columns = names
 # drop rows with NaN values
 if dropna:
    agg.dropna(inplace=True)
 return agg
#
# ## cold season
#values=DF_var_cold.values

### warm season
values=DF_var_warm.values
# # integer encode direction
# encoder = LabelEncoder()
# values[:,4] = encoder.fit_transform(values[:,4])

# ensure all data is float
values = values.astype('float64')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = StandardScaler()
scaled = scaler.fit_transform(values)

# # frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# split into train and test sets
values = reframed.values

n_train_hours = 8760
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
tf.data.experimental.enable_debug_mode()
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 9:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]*1000
# # invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 9:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]*1000
# # calculate RMSE
rmse = round(100* sqrt(mean_squared_error(inv_y, inv_yhat))/np.max(inv_y) , 2)
print('Test RMSE(%): ',rmse)