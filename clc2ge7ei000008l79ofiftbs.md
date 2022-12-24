# Time series forecasting of stock data using a deep learning model.

To forecast stock data using a deep learning model, we will follow the following steps:

1. Collect and pre-process the data: We will first need to collect the stock data for the time period we want to forecast. This can be done by accessing financial databases or by manually collecting the data from sources such as stock exchange websites. Next, we will pre-process the data by cleaning and normalizing it. This may include removing any missing or corrupted data, as well as scaling the data to make it easier for the model to process.
    
2. Build the deep learning model: Once the data has been pre-processed, we will build the deep learning model using a neural network architecture. This may include selecting the type of model (such as a recurrent neural network or a convolutional neural network) and determining the number and size of the layers. We will also need to determine the optimal hyperparameters for the model, such as the learning rate and the number of epochs.
    
3. Train the model: Once the model has been built, we will train it using the pre-processed data. This will involve feeding the data into the model and adjusting the weights and biases to optimize the model's performance.
    
4. Test the model: After the model has been trained, we will need to test its performance on a separate dataset to ensure that it is able to accurately predict future stock prices.
    
5. Make predictions: Once the model has been trained and tested, we can use it to make predictions on future stock data. This may involve inputting new data into the model and using the output to make informed decisions about buying and selling stocks.
    

As an example, let's say we want to forecast the stock price of Company X for the next month using a deep learning model. Here are the steps we would follow:

1. Collect and pre-process the data: We collect the stock data for Company X for the past year and pre-process it by cleaning and normalizing the data.
    
2. Build the deep learning model: We decide to use a recurrent neural network as our model, with two hidden layers and a learning rate of 0.001. We also determine that we will train the model for 50 epochs.
    
3. Train the model: We feed the pre-processed data into the model and train it using the specified hyperparameters.
    
4. Test the model: We test the model's performance on a separate dataset and find that it can accurately predict stock prices with an error rate of 2%.
    
5. Make predictions: We input new data into the model and use the output to make informed decisions about buying and selling Company X stocks in the next month.
    

Here is an example of code that can be used to forecast stock data using a deep-learning model with CSV data:

First, we will import the necessary libraries and read the CSV data:

You will find this data in the Kaggle dataset in the following link [Stock Market daily data](https://www.kaggle.com/datasets/paultimothymooney/stock-market-data)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Read in the CSV data
df = pd.read_csv('stock_data.csv')
```

Next, we will pre-process the data by cleaning and normalizing it:

```python
# Convert the 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Extract the year, month, and day as separate columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Drop the original 'Date' column
df = df.drop(columns=['Date'])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Split the data into training and testing sets
train_size = int(len(df_scaled) * 0.8)
test_size = len(df_scaled) - train_size
train, test = df_scaled[0:train_size,:], df_scaled[train_size:len(df_scaled),:]

# Convert the data into a 3D array (a sequence with t timesteps and d dimensions)
def create_sequences(data, t, d):
    X, y = [], []
    for i in range(len(data)-t-1):
        a = data[i:(i+t), :]
        X.append(a)
        y.append(data[i + t, :])
    return np.array(X), np.array(y)

# Create sequences of t timesteps with d dimensions
t = 10 # timesteps
d = 9 # dimensions (including year, month, and day)
X_train, y_train = create_sequences(train, t, d)
X_test, y_test = create_sequences(test, t, d)
```

Then, we will build and train the deep learning model:

```python
# Build the model
model = Sequential()
model.add(LSTM(50, input_shape=(t, d)))
model.add(Dense(d))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(X_train, 
                    y_train, 
                    epochs=50, 
                    batch_size=1, 
                    verbose=1
                    )
```

Finally, we will test the model and make predictions:

```python
# Test the model
test_error = model.evaluate(X_test, y_test, verbose=2)
print(f'Test error: {test_error}')
print(f'Accuracy: {(1-test_error) * 100}%')
```

# Hope you liked it!

# Sharing knowledge and love...