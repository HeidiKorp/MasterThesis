import pandas as pd
import numpy as np
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# from tf.keras.experimental import PeepholeLSTMCell
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from math import floor

from .helper import blockify

# https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37
# https://www.w3schools.com/python/python_classes.asp
# https://keras.io/api/models/model/

class Model:

    def __init__(self, X, y,
                    dropout, 
                    recurrent_width, 
                    input_width,
                    lr,
                    train_size,
                    validation_size,
                    test_size,
                    network_length,
                    epochs):
        if train_size < 1:
            train_size = floor(len(X) * train_size)
            validation_size = floor(len(X) * validation_size)
            test_size = floor(len(X) * test_size)
        print("Train: ", train_size, " val: ", validation_size, " test: ", test_size)
        print("Train shape: ", X.shape)
        self.dropout = dropout
        self.input_width = input_width
        self.lr = lr
        self.recurrent_width = recurrent_width
        self.val_size = validation_size
        self.train_size = train_size
        self.network_length = network_length
        self.epochs = epochs
        # Make a new function to split the data
        # It must be done based on the track

        self.X, self.y = self.reshapeInput(X, y, network_length)
        print("Shapee: ", self.X.shape)
        print("Shapee3: ", self.y.shape)
        print("Actual: ", self.X)
        print("")
        print("")
        print(len(self.X))
        for i in self.X:
            print(len(i))

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        #         X[1:], y[1:], train_size=train_size+validation_size, test_size=test_size, random_state=42)

        
        # TODO Create model
        self.model = None

    def getModel(self):
        model = Sequential()

        # add model layers
        model.add(Dense(256, activation='relu', input_shape=self.X_train.shape))
        model.add(LSTM(512, activation='relu', return_sequences=True, input_shape=(5, 4)))
        model.add(LSTM(512, activation='relu', return_sequences=True, input_shape=(5, 4)))
        model.add(LSTM(512, activation='relu', input_shape=(5, 4)))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))  # 3 is the number of classes

        # compile model using mse as a measure of model performance
        opt = Adam(learning_rate=self.lr)
        model.compile(optimizer=opt, loss='mean_squared_error')
        # model.summary()
        return model


    def setModel(self):
        self.model = self.getModel()

    def train(self):
        # Preprocess the input data
        self.reshapeInput()

        history = self.model.fit(self.X_train, 
                self.y_train, 
                epochs=20, 
                validation_data=(self.X_val, self.y_val),
                batch_size=5)
        return history


    def predict(self):
        return self.model.predict(self.X_test)


    def reshapeInput(self, X, y, k):
        # k = network length
        # split data based on self.network_length (5, 10, 25)
        # the result should be a 3d data
        # n samples (tracks), net_len samples (grouped data in track), features
        col_names = list(X.columns)
        nr_tracks = len(X.ObjectId.unique())
        idx_obj_id = col_names.index('ObjectId')

        # print("X here: \n", X)

        X = X.to_numpy()
        y = y.to_numpy()
        new_X = []
        new_y = []
        counter = 0
        prev_id = -1

        # Sort by tracks and if the nr of data points in track does not divide by 5,
        # apply padding by repeating the last known data point

        for i in range(len(X)):
            print("Prev id is: ", prev_id)
            print("New X: ", new_X)
            if counter == 0:
                new_X.append([X[i][:idx_obj_id]])
                new_y.append([y[i]])
                counter += 1
                prev_id = X[i][idx_obj_id]
            elif prev_id != X[i][idx_obj_id]:
                new_X[-1].append(new_X[-1][-1])
                new_y[-1].append(new_y[-1][-1])
                
                if counter == k - 1: counter = 0
                else: counter += 1
                

        for i in range(len(X)):
            if i == 0:
                new_X.append([])
                new_y.append([])
            if counter >= k:
                new_X.append([])
                new_y.append([])
                counter = 0
            new_X[-1].append(X[i][:idx_obj_id])
            new_y[-1].append(y[i])
            counter +=1

        # for i in range(len(X)):
        #     if X[i][idx_obj_id] != prev_id:
        #         new_X.append([X[i][:idx_obj_id]])
        #         new_y.append([y[i]])
        #         prev_id = X[i][idx_obj_id]
        #     else:
        #         new_X[-1].append(X[i][:idx_obj_id])
        #         new_y[-1].append(y[i])
        return np.array(new_X), np.array(new_y)


    # def evaluate():
    #     print("Evaluate on test data")
    #     results = self.model.evaluate(self.X_test, self.y_test)
    #     print("test loss, test acc: ", results)

    #     # Generate predictions (probabilities -- the output of the last layer)
    #     # on new data using 'predict'

    #     print("Generate predictions for 3 samples")
    #     predictions = model.predict(x_test[:3])
    #     actual = self.y_test[:3]
    #     print("Predicted: ", predictions)
    #     print("Actual: ", actual)



