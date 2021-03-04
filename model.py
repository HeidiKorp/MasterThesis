import pandas as pd
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, RNN, StackedRNNCells, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from math import floor
import matplotlib.pyplot as plt

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

        inputs = Input(shape=(self.network_length, n_features,))
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
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        model.summary()

        return model


    def set_model(self):
        self.model = self.compile_model_functional()


    def train(self):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        mc = ModelCheckpoint("models/best_model_destination_"+ str(self.network_length) +  ".h5", monitor='val_accuracy', 
                            mode='max', verbose=1, save_best_only=True)
        history = self.model.fit(
                self.X_train, 
                self.y_train, 
                epochs=200, 
                validation_data=(self.X_val, self.y_val),
                callbacks=[es, mc]
                )

        hist_df = pd.DataFrame(history.history)
        hist_json_file = "models/history_" + str(self.network_length) + ".json"
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)
        return history


    def predict(self, model, n_samples):
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


    def evaluate(self, model, history):
        # print("Evaluate on test data")
        # results = model.evaluate(self.X_test, self.y_test)
        # print("test loss, test acc: ", results)

        # evaluate the model
        _, train_acc = model.evaluate(x=self.X_train, y=self.y_train, verbose=0)
        _, test_acc = model.evaluate(x=self.X_test, y=self.y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        print("Loss: ", history['loss'])
        print("Val loss: ", history['val_loss'])
        # plot training history
        plt.plot(history['loss'], label='train')
        plt.xlabel("timesteps", fontsize=18)
        plt.ylabel("loss", fontsize=18)
        plt.legend()
        plt.savefig("models/train_loss_dest_" + str(self.network_length) + ".jpg")
        plt.plot(history['val_loss'], label='test')
        plt.xlabel("timesteps", fontsize=18)
        plt.ylabel("val_sloss", fontsize=18)
        plt.legend()
        plt.savefig("models/val_loss_dest_" + str(self.network_length) + ".jpg")
        plt.close()
        # plt.legend()
        # plt.show()


    def get_best_saved_model(self):
        return load_model("models/best_model_destination_" + str(self.network_length) + ".h5")

    def get_history(self):
        return pd.read_json("models/history_" + str(self.network_length) + ".json", orient='records')