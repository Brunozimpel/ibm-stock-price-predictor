from sklearn.preprocessing import StandardScaler
from models import model1,model2
import pandas as pd
import numpy as np
import quandl
import os
import sys

# Valid API Key
# Get environment variables
quandl.ApiConfig.api_key = os.environ.get('QUANDL_API_KEY')

# Import the data automatically and setting 'date' as index
df = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['date','adj_close']}, ticker = 'IBM',paginate=True)
df = df.set_index('date')

# Consider data only from 2013 onwards (roughly 5 years)
# Considering 2013 -5 days for moving average calculation 
newest_df = df[:'2012-12-26']
# Inverting order for features generation on correct order
newest_df = newest_df[::-1]

# Create two new features, one for days and another for a moving average of
# adj_close of window's size 5
newest_df['days'] = (((newest_df.index - newest_df.index[0]).days.values)-7)
newest_df['average'] = newest_df['adj_close'].rolling(window=5).mean()

# Finally considering only 2013 onwards (to make sure there are not any NaN
# values on the average column)
# Inverting order again to filter only 2013 onwards on the corret order
newest_df = newest_df[::-1]
newest_df = newest_df[:'2013']
# Invert for the last time to train using oldest data and test using new data
newest_df = newest_df[::-1]

# Scale the features so each one of them has the same scale without distorting
# differences in the range of values StandardScaler remove the mean and scale to
# unit variance
scaler = StandardScaler()

X = newest_df[['days','average']]
scaler.fit(X)
X = scaler.transform(X)

Y = newest_df['adj_close']
scaler.fit(np.array(Y).reshape(-1,1)) # Reshape necessary to avoid errors on the train_test_split() function
Y = scaler.transform(np.array(Y).reshape(-1,1))

# User choose which model he/she wants to evaluate now
# The split between train, validation and test sets depends on the model chosen
# Split data on train, validation and test sets. 60% for training, 20% for validation and 20% for testing
# The validation proccess for fine tuning of the hyperparameters is described at the report.
# Only the optimal values for the hyperparameters were used at these models

def menu():
    print("************IBM Stock Price Predictor**************")
    print()

    choice = input("""
                1: Model 1 - Random Forest Regressor
                2: Model 2 - Long Short-Term Memory Neural Network
                Please enter your choice: """)

    if choice == "1":
        model1(X,Y)
    elif choice == "2":
        model2(X,Y)
    else:
        print("You must only select either 1 or 2 \nPlease try again")
        sys.exit(1)

menu()
