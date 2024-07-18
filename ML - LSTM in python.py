
LSTM
----------


# import required modules
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import matplotlib.pyplot as plt
import math
import seaborn as sns
import warnings

## Mount your Drive into Google Colab
from google.colab import drive
drive.mount('/content/drive')

data_INFY = pd.read_csv("/content/drive/MyDrive/Infosys.csv")

data_INFY

#Date column in new format yyyy-mm-dd to change it to datetime data type and facilitate indexing
Date = data_INFY['Date ']
New_Date =[]
dd =''
mm_name =''
mm =''
yy =''
for dt in Date:
    dt = str(dt)
    dd = dt[:2]
    mm_name = dt[3:6]
    yyyy = dt[7:]
    if(mm_name == 'JAN' or mm_name == 'Jan'):
        mm = '01'
    elif(mm_name == 'FEB' or mm_name == 'Feb'):
        mm = '02'
    elif(mm_name == 'MAR'or mm_name == 'Mar'):
        mm = '03'
    elif(mm_name == 'APR'or mm_name == 'Apr'):
        mm = '04'
    elif(mm_name == 'MAY'or mm_name == 'May'):
        mm = '05'
    elif(mm_name == 'JUN'or mm_name == 'Jun'):
        mm = '06'
    elif(mm_name == 'JUL'or mm_name == 'Jul'):
        mm = '07'
    elif(mm_name == 'AUG'or mm_name == 'Aug'):
        mm = '08'
    elif(mm_name == 'SEP'or mm_name == 'Sep'):
        mm = '09'
    elif(mm_name == 'OCT'or mm_name == 'Oct'):
        mm = '10'
    elif(mm_name == 'NOV'or mm_name == 'Nov'):
        mm = '11'
    elif(mm_name == 'DEC'or mm_name == 'Dec'):
        mm = '12'
    Dt = yyyy + '-' + mm + '-' + dd
    New_Date.append(Dt)
data_INFY['Trading_Date'] = New_Date

data_INFY

#Convert Open, High, Low, Close columns from string to float data type
Open = data_INFY['OPEN ']
Float_Open =[]
for each in Open:
    each = each.replace(',', '')
    Float_Open.append(each)
data_INFY['OPEN'] = Float_Open
data_INFY['OPEN'] = data_INFY['OPEN'].astype(float)

High = data_INFY['HIGH ']
Float_High =[]
for each in High:
    each = each.replace(',', '')
    Float_High.append(each)
data_INFY['HIGH'] = Float_High
data_INFY['HIGH'] = data_INFY['HIGH'].astype(float)

Low = data_INFY['LOW ']
Float_Low =[]
for each in Low:
    each = each.replace(',', '')
    Float_Low.append(each)
data_INFY['LOW'] = Float_Low
data_INFY['LOW'] = data_INFY['LOW'].astype(float)

Close = data_INFY['close ']
Float_Close =[]
for each in Close:
    each = each.replace(',', '')
    Float_Close.append(each)
data_INFY['CLOSE'] = Float_Close
data_INFY['CLOSE'] = data_INFY['CLOSE'].astype(float)

data_INFY

data_INFY['Trading_Date'] = pd.to_datetime(data_INFY['Trading_Date'])
data_INFY = data_INFY.sort_values(by='Trading_Date')

data_INFY = data_INFY.drop(labels=['Date ', 'series ', 'OPEN ', 'HIGH ', 'LOW ', 'PREV. CLOSE ', 'ltp ', 'close ', 'vwap ', '52W H ', '52W L ', 'VOLUME ', 'VALUE ', 'No of trades '], axis=1)

data_INFY

## Training the model with more columns

open_prev_close = list(data_INFY['OPEN'][1:].values - data_INFY['CLOSE'][:-1].values)
open_prev_close.append(np.nan)

data_INFY = data_INFY.assign(Open_Diff = data_INFY['OPEN'].diff(),
                   High_Diff = data_INFY['HIGH'].diff(),
                   Low_Diff = data_INFY['LOW'].diff(),
                   Close_Diff = data_INFY['CLOSE'].diff(),
                   Open_Close_Diff = data_INFY['OPEN'] - data_INFY['CLOSE'],
                   Open_Low_Diff = data_INFY['OPEN'] - data_INFY['LOW'],
                   High_Low_Diff = data_INFY['HIGH']- data_INFY['LOW'],
                   High_Open_Diff = data_INFY['HIGH'] - data_INFY['OPEN'],
                   Close_Low_Diff = data_INFY['CLOSE'] - data_INFY['LOW'],
                   Close_High_Diff = data_INFY['CLOSE'] - data_INFY['HIGH'],
                   Open_PrevOpen_Diff = data_INFY['OPEN'].diff().shift(-1),
                   Open_PrevClose_Diff = open_prev_close)

data_INFY

data_INFY = data_INFY.dropna()
data_INFY.shape[0]

## spliiting the 6 months data for training and testiong
## Training: Trading date from June to sept
## Testing: October to December

time_mask1 = (data_INFY['Trading_Date'].dt.month >= 6) & (data_INFY['Trading_Date'].dt.month <10) & (data_INFY['Trading_Date'].dt.year >= 2021)
Training_Data_INFY = data_INFY[time_mask1]
Training_Data_INFY = Training_Data_INFY.set_index('Trading_Date')
Training_Data_INFY

time_mask2 = (data_INFY['Trading_Date'].dt.month >= 10) & (data_INFY['Trading_Date'].dt.year == 2022)
Testing_Data_INFY = data_INFY[time_mask2]
Testing_Data_INFY = Testing_Data_INFY.set_index('Trading_Date')
Testing_Data_INFY

## For normalization of data

sc_train = StandardScaler()
training_set_scaled = sc_train.fit_transform(Training_Data_INFY)

sc_test = StandardScaler()
testing_set_scaled = sc_test.fit_transform(Testing_Data_INFY)

#Train data for predicting High price for call option
X_train = []
y_train = []
for i in range(5, training_set_scaled.shape[0]): ## batch size=5
    X_train.append(training_set_scaled[i-5:i, 0:16])
    y_train.append(training_set_scaled[i, 1])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 16))

X_train

y_train

#Mean Directional Accuracy
def mda(actual: np.ndarray, predicted: np.ndarray):
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - actual[:-1])).astype(int))

## Reshaping the test data

sc = StandardScaler()
dataset_test = pd.DataFrame().assign(High = Testing_Data_INFY['HIGH'])
inputs = dataset_test[:].values
inputs = inputs.reshape(-1,1)
inputs1 = sc.fit_transform(inputs)
X_test = []
y_test = []
for i in range(5, len(testing_set_scaled)):
    X_test.append(testing_set_scaled[i-5:i, 0:16])
for i in range(5, len(dataset_test)):
    y_test.append(inputs[i, 0])
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 16))
X_test

y_test ## WE want to test for High column of test data

regressor = Sequential()

regressor.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train.shape[1], 16)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 96, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 64))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 1000, batch_size = None, shuffle = True)

## to get the original value from the scaled value

y_pred = []
y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)
y_pred

## mapping indexes for y-pred and y_test

Testing_Data_INFY_for_index = Testing_Data_INFY.iloc[5:]
y_pred_indexed = pd.DataFrame(y_pred).set_index(Testing_Data_INFY_for_index.index)
y_test_indexed = pd.DataFrame(y_test).set_index(Testing_Data_INFY_for_index.index)

y_pred_indexed

plt.rcParams.update({'figure.figsize' : (7.5,2), 'figure.dpi' : 65 , 'axes.edgecolor' : "black", 'axes.facecolor': 'white' })
plt.plot(y_test_indexed, color = 'Red', label = 'Actual High Price')
plt.plot(y_pred_indexed, color = 'Blue', label = 'LSTM Model Predicted High Price')

plt.title('Infosys High Price Prediction - LSTM Model', fontsize = 14,  color = 'black')
plt.xlabel('Trading Date', fontsize = 14, color = 'black')
plt.ylabel('High Price (in Rs)', fontsize = 14, color = 'black')
leg =plt.legend(fontsize =10)

plt.xticks(fontsize=12, color = 'black')
plt.yticks(fontsize=12, color = 'black')
plt.xticks(rotation = 30)
for text in leg.get_texts():
    text.set_color("black")
plt.show()

# report performance
directional_accuracy = mda(y_test ,y_pred)
print('Directional Accuracy: ' + str(directional_accuracy))
mse = mean_squared_error(y_test, y_pred)
print('MSE: '+str(mse))
mae = mean_absolute_error(y_test, y_pred)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(y_pred - y_test)/np.abs(y_test))
print('MAPE: '+str(mape))

