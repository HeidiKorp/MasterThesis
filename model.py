import random
import pandas as pd
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, RNN, StackedRNNCells, Input, Embedding, GRU, SimpleRNN
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from math import floor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random

from helper import get_train_val_test, split_sequences, oneHotEncode, stepsToOne, reshapeData, codeToDest, normalizeData, split_tracks, evenNrDatapoints

# https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# https://www.w3schools.com/python/python_classes.asp
# https://keras.io/api/models/model/
# This model predicts the exit of the vehicle
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
                    epochs,
                    fileName,
                    newData):
        self.dropout = dropout
        self.input_width = input_width
        self.lr = lr
        self.recurrent_width = recurrent_width
        self.epochs = epochs
        self.network_length = network_length
        self.fileName = fileName
        
        # First onehot
        # Train/val/test
        # Reshape
        # Shuffle

        y_col_nr = len(codeToDest)

        if newData:
            # One-hot encode the y values
            # onehots = oneHotEncode(data[['destination']])
            # data = data.drop(['codeBin', 'destination'], axis=1)
            onehots = oneHotEncode(data[['code']])
            data = data.drop(['code'], axis=1)
            # data = normalizeData(data)
            data = pd.concat([data, onehots], axis=1)
            print("Data cols: \n", data.columns)

            ids = data.uniqueId.unique()
            random.shuffle(ids)
            print(ids)
            train_len = int(len(ids) * train_size)
            val_len = int(len(ids) * validation_size)
            print("Train len: ", train_len)
            print("Val len: ", val_len)
            
            train_ids = ids[:train_len]
            val_ids = ids[train_len:train_len+val_len]
            test_ids = ids[train_len+val_len:]

            self.train_data = data.loc[data.uniqueId.isin(train_ids)]
            self.val_data = data.loc[data.uniqueId.isin(val_ids)]
            self.test_data = data.loc[data.uniqueId.isin(test_ids)]

            # self.train_data, self.val_data, self.test_data = \
            # get_train_val_test(data, train_size, validation_size, test_size)

            self.train_data.to_csv("additions/datasets/april/train.csv")
            self.val_data.to_csv("additions/datasets/april/validation.csv")
            self.test_data.to_csv("additions/datasets/april/test.csv")

        self.train_data = pd.read_csv("additions/datasets/april/train.csv", dtype='category')
        self.val_data = pd.read_csv("additions/datasets/april/validation.csv", dtype='category')
        self.test_data = pd.read_csv("additions/datasets/april/test.csv", dtype='category')
        print("Read data files!")

        # Even out the tracks
        # print("TRain data shape: ", self.train_data.shape)
        self.train_data = evenNrDatapoints(self.train_data, self.network_length)
        print("Train file cols: ", self.train_data.columns)
        print("Evened train file!")
        self.val_data = evenNrDatapoints(self.val_data, self.network_length)
        print("Evened val file")
        self.test_data = evenNrDatapoints(self.test_data, self.network_length)
        print("Evened test file")
        print("X train shape: ", self.train_data.shape)

        # Save evened-out datasets
        self.train_data.to_csv("additions/datasets/april/train-" + str(self.network_length) + ".csv", index=False)
        self.val_data.to_csv("additions/datasets/april/validation-" + str(self.network_length) + ".csv", index=False)
        self.test_data.to_csv("additions/datasets/april/test-" + str(self.network_length) + ".csv", index=False)

        # # Read saved evened-out datasets
        # self.train_data = pd.read_csv("additions/datasets/april/train-" + str(self.network_length) + ".csv", dtype='category')
        # self.val_data = pd.read_csv("additions/datasets/april/validation-" + str(self.network_length) + ".csv", dtype='category')
        # self.test_data = pd.read_csv("additions/datasets/april/test-" + str(self.network_length) + ".csv", dtype='category')

        self.train_data = self.train_data[self.train_data.columns.drop(list(self.train_data.filter(regex='Unnamed')))]
        self.val_data = self.val_data[self.val_data.columns.drop(list(self.val_data.filter(regex='Unnamed')))]
        self.test_data = self.test_data[self.test_data.columns.drop(list(self.test_data.filter(regex='Unnamed')))]

        self.train_data = self.train_data.drop(['uniqueId'], axis=1)
        self.val_data = self.val_data.drop(['uniqueId'], axis=1)
        self.test_data = self.test_data.drop(['uniqueId'], axis=1)

        self.train_data = self.train_data.to_numpy()
        self.val_data = self.val_data.to_numpy()
        self.test_data = self.test_data.to_numpy()

        # print(self.train_data.columns)

        # # print("Shape x train: ", self.train_data.shape)
        # # print(type(self.train_data))
        # # # print(self.train_data[:3])
        # # print(self.train_data[:3, :3])

        self.X_train, self.y_train = self.train_data[:, :-y_col_nr-1], self.train_data[:, -y_col_nr:]
        self.X_val, self.y_val = self.val_data[:, :-y_col_nr-1], self.val_data[:, -y_col_nr:]
        self.X_test, self.y_test = self.test_data[:, :-y_col_nr-1], self.test_data[:, -y_col_nr:]
        
        print(self.X_train[:5])
        print(self.y_train[:5])
        # Numpy reshape
        self.X_train = reshapeData(self.X_train, self.network_length)
        self.y_train = reshapeData(self.y_train, self.network_length)

        self.X_val = reshapeData(self.X_val, self.network_length)
        self.y_val = reshapeData(self.y_val, self.network_length)

        self.X_test = reshapeData(self.X_test, self.network_length)
        self.y_test = reshapeData(self.y_test, self.network_length)
        
        print("X train shape: ", self.X_train.shape)
        print("y train shape: ", self.y_train.shape)
        # # self.X_train, self.y_train = self.reshape_input(self.train_data, network_length)
        # # self.X_val, self.y_val = self.reshape_input(self.val_data, network_length)
        # # self.X_test, self.y_test = self.reshape_input(self.test_data, network_length)
       
        # self.X_train, self.y_train = shuffle(np.array(self.X_train), np.array(self.y_train))
        # self.X_val, self.y_val = shuffle(np.array(self.X_val), np.array(self.y_val))
        # self.X_test, self.y_test = shuffle(np.array(self.X_test), np.array(self.y_test))

        # # print("Val head: \n", self.X_val[:3])

        # # print("TrainX: \n", type(self.X_train))
        # # print(self.X_train[:3])
        # # print("Train y:\n", type(self.y_train))
        # # print(self.y_train[:3])
        
        # # X = data.to_numpy()
        # # X = X[:,:-1]
        # # print("X shape: ", X.shape)
        # # print("Len X: ", len(X))
        # # a = X.shape[0] // self.network_length
        # # b = a * self.network_length
        # # X = np.reshape(X[:b], (a, self.network_length, X.shape[1]))
        # # y = np.reshape(onehots[:b], (a, self.network_length, onehots.shape[1]))
        # # X, y = shuffle(np.array(X), np.array(y))
        # # print("Shape X: \n", X.shape)

        # # print("Hots: \n", onehots.head())
        # # data = pd.concat([data.reset_index(), onehots.reset_index()], axis=1)


        # # print("Data head:\n", data['west'].head())
        # # data = data.drop(['codeBin', 'destination'], axis=1)
        # # print("Data head:\n", data.head())
        # # X, y = self.reshape_input(data, network_length)
        # # X, y = shuffle(np.array(X), np.array(y))
        # # X, y = stepsToOne(X, y)

        # # print("X firsts: \n", X[:3])
        # # print("Y firsts: \n", y[:3])
        # # l = list(reshaped)
        # # print("First el: \n", l[0])
        # # random.shuffle(l)
        # # tup = tuple(l)
        # # X, y = tup
        # # print("Y is: \n", y[:3])
        # # print("Tup:\n", tup[:3])
        # # shuffled = reshaped.sample(frac=1).reset_index()
        # # Split the data to train, validation and test sets

        # # self.X_train, self.X_test, self.y_train, self.y_test \
        # #         = train_test_split(X, y, test_size=test_size, random_state=1)
        # # self.X_train, self.X_val, self.y_train, self.y_val \
        # #         = train_test_split(self.X_train, self.y_train, test_size=validation_size, random_state=1)

        self.X_train=np.asarray(self.X_train).astype(np.float)
        self.y_train=np.asarray(self.y_train).astype(np.float)

        self.X_val=np.asarray(self.X_val).astype(np.float)
        self.y_val=np.asarray(self.y_val).astype(np.float)

        self.X_test=np.asarray(self.X_test).astype(np.float)
        self.y_test=np.asarray(self.y_test).astype(np.float)

        
        # # self.train_data, self.val_data, self.test_data = \
        # #     get_train_val_test(tup, train_size, validation_size, test_size)
        # # self.X_train, self.y_train = self.train_data
        # # self.X_val, self.y_val = self.val_data
        # # self.X_test, self.y_test = self.test_data

        # # self.y_train = oneHotEncode(self.y_train)
        # # self.y_val = oneHotEncode(self.y_val)
        # # self.y_test = oneHotEncode(self.y_test)
        # # # Reshape the data based on network length (5, 15, 25)
        # # self.X_train, self.y_train = self.reshape_input(self.train_data, network_length)
        # # self.X_val, self.y_val = self.reshape_input(self.val_data, network_length)
        # # self.X_test, self.y_test = self.reshape_input(self.test_data, network_length)
        self.model = None


    def get_model(self):
        n_features = self.X_train.shape[1]
        model = Sequential()
        
        # add model layers
        model.add(Dense(256, activation='relu', input_shape=(self.network_length, n_features)))
        model.add(LSTM(512, activation='relu', return_sequences=True, input_shape=(self.network_length, n_features)))
        model.add(LSTM(512, activation='relu', return_sequences=True, input_shape=(self.network_length, n_features)))
        model.add(LSTM(512, activation='relu', input_shape=(self.network_length, n_features)))
        # model.add(Dropout(0.5))
        model.add(Dense(8, activation='softmax'))  # 3 is the number of classes

        # compile model using mse as a measure of model performance
        opt = Adam(learning_rate=self.lr)
        model.compile(optimizer=opt, loss='mean_squared_error')
        model.summary()
        # return model

        
    def compile_model_functional(self):
        n_features = self.X_train.shape[2]

        inputs = Input(shape=(self.network_length, n_features,))
        dense = Dense(256, activation='relu', input_shape=(self.network_length, n_features,))(inputs)
        cells = [tfa.rnn.PeepholeLSTMCell(512, activation='relu') for _ in range(1)]
        # Wrap the Peephole cells with RNN and set return_sequences=True
        rnn = tf.keras.layers.RNN(cells, return_sequences=True)(dense)
        # lstm = LSTM(512, activation='relu', return_sequences=True)(dense)
        # dropout = Dropout(0.5)(rnn)
        outputs = tf.keras.layers.Dense(8, activation='softmax')(rnn)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Optimizer and loss are not defined!
        # compile model using mse as a measure of model performance
        opt = Adam(learning_rate=self.lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        # n_features = self.X_train.shape[2]
        # model = Sequential()
        # model.add(Dense(256, activation='relu', input_shape=(self.network_length, n_features,)))

        # # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
        # model.add(LSTM(256, return_sequences=True))

        # # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
        # # model.add(SimpleRNN(128))
        # # model.add(Dropout(0.2))

        # model.add(Dense(8))
        # # Optimizer and loss are not defined!
        # # compile model using mse as a measure of model performance
        # opt = Adam(learning_rate=self.lr)
        # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # model.summary()
        # https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/
        # https://stackoverflow.com/questions/54138205/shaping-data-for-lstm-and-feeding-output-of-dense-layers-to-lstm
        
        # n_features = self.X_train.shape[2]

        # inputs = Input(shape=(self.network_length, n_features,))
        # dense = Dense(256, activation='relu', input_shape=(self.network_length, n_features,))(inputs)
        # norm = BatchNormalization()(dense)
        # # Create a stack of LSTM with Peephole connection layers
        # cells = [tfa.rnn.PeepholeLSTMCell(512, activation='relu') for _ in range(3)]
        # # Wrap the Peephole cells with RNN and set return_sequences=True
        # rnn = tf.keras.layers.RNN(cells, return_sequences=True)(norm)
        # dropout = Dropout(0.5)(rnn)
        # outputs = tf.keras.layers.Dense(8, activation='softmax')(dropout)
        # model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # # Optimizer and loss are not defined!
        # # compile model using mse as a measure of model performance
        # opt = Adam(learning_rate=self.lr)
        # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        # model.summary()

        return model


    def set_model(self):
        self.model = self.compile_model_functional()


    def train(self):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        mc = ModelCheckpoint("additions/datasets/april/" + self.fileName + "_" + str(self.network_length) + "/best_model_destination_"+ str(self.network_length) +  ".h5", monitor='val_accuracy', 
                            mode='max', verbose=1, save_best_only=True, save_freq='epoch')
        # print("X train: ", self.X_train[:3])
        # print("Y train: ", self.y_train[:3])
        # print("Shape of X: ", self.X_train.shape, " y shape: ", self.y_train.shape)
        history = self.model.fit(
                self.X_train, 
                self.y_train, 
                epochs=self.epochs, 
                validation_data=(self.X_val, self.y_val),
                callbacks=[es, mc]
                )

        hist_df = pd.DataFrame(history.history)
        hist_json_file = "additions/datasets/april/" + self.fileName + "_" + str(self.network_length) + "/history_" + str(self.network_length) + ".json"
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)
        return history


    def predict(self, model, n_samples):
        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using 'predict'
        print("Generate predictions for %s samples" % (n_samples))
        predictions = model.predict(self.X_test[:n_samples])
        # actual = self.y_test[:n_samples]
        # print("Predicted: ", predictions)
        # print("Actual: ", actual)
            # Converting predictions to label
        pred = list()
        for i in range(len(predictions)):
            pred.append(np.argmax(y_pred[i]))

        # Converting one hot encoded test label to label
        test = list()
        for i in range(len(self.y_test[:n_samples])):
            test.append(np.argmax(self.y_test[:n_samples][i]))
        
        print("Result: ", test[4895:4900])
        print("Y_test: ", pred[4895:4900])
        # Get accuracy
        a = accuracy_score(test, pred)
        print("Accuracy is: ", a * 100)


    def reshape_input(self, X, k):
        # Get the index of OdjectId
        col_names = list(X.columns)
        idx_obj_id = col_names.index('uniqueId')
        # Convert pandas dataFrame to numpy array
        X = X.to_numpy()
        return split_sequences(X, k, idx_obj_id)


    def evaluateClass(self, model, history, data):
        data = data[data.columns.drop(list(data.filter(regex='Unnamed')))]
        data.drop(['uniqueId'], axis=1)
        print(data.columns)


    def evaluateNoPlot(self, model, history, X_test, Y_test):
        _, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        print("Test acc: %.3f" % (test_acc))
        print("Loss: ", history['loss'])
        print("Val loss: ", history['val_loss'])



    def evaluate(self, model, history, data):
        # print("Evaluate on test data")
        # results = model.evaluate(self.X_test, self.y_test)
        # print("test loss, test acc: ", results)

        # evaluate the model
        # _, train_acc = model.evaluate(x=self.X_train, y=self.y_train, verbose=0)
        _, test_acc = model.evaluate(x=self.X_test, y=self.y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        print("Loss: ", history['loss'])
        print("Val loss: ", history['val_loss'])
        # plot training history
        plt.plot(history['loss'], label='train')
        plt.xlabel("timesteps", fontsize=18)
        plt.ylabel("loss", fontsize=18)
        plt.legend()
        plt.savefig("additions/datasets/april/" + self.fileName + "_" + str(self.network_length) + "/train_loss_dest_" + str(self.network_length) + ".jpg")
        plt.plot(history['val_loss'], label='test')
        plt.xlabel("timesteps", fontsize=18)
        plt.ylabel("val_sloss", fontsize=18)
        plt.legend()
        plt.savefig("additions/datasets/april/" + self.fileName + "_" + str(self.network_length) + "/val_loss_dest_" + str(self.network_length) + ".jpg")
        plt.close()
        # plt.legend()
        # plt.show()


    def get_best_saved_model(self):
        return load_model("additions/datasets/april/" + self.fileName + "_" + str(self.network_length) + "/best_model_destination_" + str(self.network_length) + ".h5")

    def get_history(self):
        return pd.read_json("additions/datasets/april/" + self.fileName + "_" + str(self.network_length) + "/history_" + str(self.network_length) + ".json", orient='records')