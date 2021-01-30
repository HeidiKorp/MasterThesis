import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding

# Layer 1: dim (k x m) 
    # k = nr of input samples
    # m = dimension of each input sample
# Layer 2: dim (m x n)


def getModel():
    model = Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(Embedding(input_dim=1000, output_dim=64))

    # Add a LSTM layer with 128 internal units.
    model.add(LSTM(128))

    # Add a Dense layer with 10 units.
    model.add(Dense(10))

    model.summary()


def getModelMine():
    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_shape=(3747, 4)))
    model.add(LSTM(units=512, return_sequences=True, activation='relu'))
    model.add(LSTM(units=521, return_sequences=True, activation='relu'))
    model.add(LSTM(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=3, activation='softmax'))
    
    model.summary()

# From: https://towardsdatascience.com/ultimate-guide-to-input-shape-and-model-complexity-in-neural-networks-ae665c728f4b
def getModel3():
    # This model consists of three hidden units and an input layer
    # Dropout is added for regularization
    # In Keras, the input dimension needs to be given excluding the batch-size (nr of samples)
    # 32 refers to the number of features in each input sample
    # Another way to give the input dimension in this model is (None, 32,)

    model = Sequential()
    model.add(Dense(units=12, activation='relu', input_shape=(32,)))
    model.add(Dropout(0.5))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='softmax'))
    model.summary()


def getRecurrentModel():
    model = Sequential()
    model.add(LSTM(32, input_shape=(100, 2)))
    model.add(Dense(1))
    model.summary()


def main():
    getModelMine()

if __name__ == "__main__":
    main()