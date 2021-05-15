from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

# Hiding unnecessary TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Model 1 - Based on Random Forest Regressor
def model1(X,Y):
    # Call the function train_test_split two times to separate the data on the 3 groups
    x_train, x_test_val, y_train, y_test_val = train_test_split(X, Y,
                                                                test_size=0.40,shuffle=False)
    x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val,
                                                    test_size=0.50,shuffle=False)

    # Reshape necessary to avoid errors on the fit() function
    y_train = np.ravel(y_train) 

    # Create a forest of 100 decision trees
    rf = RandomForestRegressor(n_estimators = 100)
    # Build forest of trees from trainig set
    rf.fit(x_train,y_train)
    # Predict regression target for x_test
    y_pred_rf = rf.predict(x_test)

    # Print main error metric of the prediction
    mse = mean_squared_error(y_test, y_pred_rf)
    print("\n##### Main results of the Random Forest Regressor algorithm #####\n")
    print("Mean Square Error of Random Forest model: %.4f" % mse)

    # Create and save figure comparing the test data with the prediction
    plt.plot(x_test[:,0],y_test,label='data')
    plt.plot(x_test[:,0],y_pred_rf,color='red',label='RF model')
    plt.xlabel("Days (scaled)")	
    plt.ylabel("adj_close (US$) (scaled)")
    plt.legend()
    plt.savefig('figures/randomForestRegressor.png', bbox_inches='tight')
    print("\nCheck randomForestRegressor.png for a visual comparison!\n")


# ----------------------------------------------------------------------------------------------------------------
# Model 2 - Based on LSTM Networks
def model2(X,Y):
    # Reshapes necessaries to avoid errors on fit() function 
    x_train, x_val_test, y_train, y_val_test = train_test_split(np.array(X).reshape(-1,2,1),
                                                                np.array(Y).reshape(-1,1),
                                                                test_size=0.40,shuffle=False)
    x_val,x_test,y_val,y_test = train_test_split(x_val_test,y_val_test,
                                                 test_size=0.50,shuffle=False)
    
    # Reshape necessary to avoid errors on the fit() function
    y_train = np.ravel(y_train) 

    # Optimal hyperparameters found on validation process
    epoch_opt = 30
    batch_opt = 64
    units_opt = 50
    dropout_prob_opt = 0.7
    optm_opt = 'nadam'

    # Create a neural network model by passing sequence of layers with Sequential() function
    model = Sequential()
    # Input LSTM layer
    model.add(LSTM(units=units_opt, return_sequences=True,
                   input_shape=(x_train.shape[1],1)))
    # Dropout layer
    model.add(Dropout(dropout_prob_opt))
    # LSTM layer
    model.add(LSTM(units=units_opt))
    # Dropout layer
    model.add(Dropout(dropout_prob_opt))
    # Fully connected output layer
    model.add(Dense(1))

    # Compile model created
    model.compile(loss='mean_squared_error', optimizer=optm_opt)

    # Train model using training dataset
    model.fit(x_train, y_train, epochs=epoch_opt, batch_size=batch_opt,
              verbose=0)
    # Predict regression target for x_test
    y_pred_lstm = model.predict(x_test)

    # Print main error metric of the prediction
    mse = mean_squared_error(y_test, y_pred_lstm)
    print("\n##### Main results of the Long Short-Term Memory Neural Network algorithm #####\n")
    print("Mean Square Error of LSTM model: %.4f" % mse)

    # Create and save figure comparing the test data with the prediction
    plt.plot(x_test[:,0],y_test,label='data')
    plt.plot(x_test[:,0],y_pred_lstm,color='red',label='LSTM model')
    plt.xlabel("Days (scaled)")	
    plt.ylabel("adj_close (US$) (scaled)")
    plt.legend()
    plt.savefig('figures/LSTMRegressor.png', bbox_inches='tight')
    print("\nCheck LSTMRegressor.png for a visual comparison!\n")
