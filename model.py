import pandas as pd
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, RNN, StackedRNNCells, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from math import floor

from helper import get_train_val_test, split_sequences

# https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# https://www.w3schools.com/python/python_classes.asp
# https://keras.io/api/models/model/

class Model:

    def __init__(self, 
                    data,
                    dropout, 
                    recurrent_width, 
                    input_width,
                    lr,
                    train_size,
                    validation_size,
                    test_size,
                    network_length,
                    epochs):
        self.dropout = dropout
        self.input_width = input_width
        self.lr = lr
        self.recurrent_width = recurrent_width
        self.epochs = epochs
        self.network_length = network_length

        # Split the data to train, validation and test sets
        self.train_data, self.val_data, self.test_data = \
            get_train_val_test(data, train_size, validation_size, test_size)
        # # Reshape the data based on network length (5, 10, 25)
        self.X_train, self.y_train = self.reshape_input(self.train_data, network_length,)
        self.X_val, self.y_val = self.reshape_input(self.val_data, network_length)
        self.X_test, self.y_test = self.reshape_input(self.test_data, network_length)

        print("X train shape: ", self.X_train.shape)
        self.model = None


    def get_model(self):
        n_features = self.X_train.shape[2]
        model = Sequential()
        
        # add model layers
        model.add(Dense(256, activation='relu', input_shape=(self.network_length, n_features)))
        model.add(LSTM(512, activation='relu', return_sequences=True, input_shape=(self.network_length, n_features)))
        model.add(LSTM(512, activation='relu', return_sequences=True, input_shape=(self.network_length, n_features)))
        model.add(LSTM(512, activation='relu', input_shape=(self.network_length, n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='softmax'))  # 3 is the number of classes

        # compile model using mse as a measure of model performance
        opt = Adam(learning_rate=self.lr)
        model.compile(optimizer=opt, loss='mean_squared_error')
        model.summary()
        # return model

        
    def compile_model_functional(self):
        # https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/
        # https://stackoverflow.com/questions/54138205/shaping-data-for-lstm-and-feeding-output-of-dense-layers-to-lstm
        
        n_features = self.X_train.shape[2]

        inputs = Input(batch_shape=self.X_train.shape)
        dense = Dense(256, activation='relu', input_shape=(self.network_length, n_features,))(inputs)
        # Create a stack of LSTM with Peephole connection layers
        cells = [tfa.rnn.PeepholeLSTMCell(512, activation='relu') for _ in range(3)]
        # Wrap the Peephole cells with RNN and set return_sequences=True
        rnn = tf.keras.layers.RNN(cells, return_sequences=True)(dense)
        dropout = Dropout(0.5)(rnn)
        outputs = tf.keras.layers.Dense(1, activation='softmax')(dropout)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # compile model using mse as a measure of model performance
        opt = Adam(learning_rate=self.lr)
        model.compile(optimizer=opt, loss='mean_squared_error')
        model.summary()

        return model


    def set_model(self):
        self.model = self.compile_model_functional()


    def train(self):
        history = self.model.fit(
                self.X_train, 
                self.y_train, 
                epochs=2, 
                validation_data=(self.X_val, self.y_val)
                )
        return history


    def predict(self, n_samples):
        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using 'predict'
        print("Generate predictions for %s samples" % (n_samples))
        predictions = model.predict(self.X_test[:n_samples])
        actual = self.y_test[:n_samples]
        print("Predicted: ", predictions)
        print("Actual: ", actual)


    def reshape_input(self, X, k):
        # Get the index of OdjectId
        col_names = list(X.columns)
        idx_obj_id = col_names.index('ObjectId')
        # Convert pandas dataFrame to numpy array
        X = X.to_numpy()
        return split_sequences(X, k, idx_obj_id)


    def evaluate(self):
        print("Evaluate on test data")
        results = self.model.evaluate(self.X_test, self.y_test)
        print("test loss, test acc: ", results)