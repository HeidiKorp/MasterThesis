import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import ast
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, RNN, StackedRNNCells, Input

best_acc = 0
best_loss = 100
best_d1 = 0
best_d2 = 0
best_epoch = 0

def normalizeCol(col_data, sc):
    orig_data = col_data.to_numpy()
    
    # Remove empty rows
    data = [x for x in orig_data if x != '[]']

    # Convert data to list of floats
    for i in range(len(data)):
        data[i] = ast.literal_eval(data[i])
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
            
    df = pd.DataFrame(data)
    scaled = sc.fit_transform(df)

    # Add back empty rows
    res = []
    next = 0
    for i in range(len(orig_data)):
        if orig_data[i] == '[]':
            res.append([])
        else:
            res.append(scaled[next])
            next += 1
    
    return res


def getNormalizedData(data, main_cols, peer_cols, save=False, outFile=None):
    data_Y = data.action
    
    data_main = data[main_cols]
    data_peers = data[peer_cols]

    data_main = data_main.astype(float)
    # Normalize the data
    sc = StandardScaler()
    val_main = sc.fit_transform(data_main)
    val_main = pd.DataFrame(data_main, columns=main_cols)

    norm_peer = pd.DataFrame(columns=peer_cols)
    for col in peer_cols:
        norm_peer[col] = normalizeCol(data_peers[col], sc)
    
    data_X = pd.concat([data_main, norm_peer], ignore_index=True, axis=1)
    cols = main_cols + peer_cols

    data_X.columns = cols
    if save:
        data_X.to_csv(outFile)
    return data_X.to_numpy(), data_Y.to_numpy()


def oneHotEncoder(data):
    # Get the categorical data to numerical representation
    le = LabelEncoder()
    integer_encoded = le.fit_transform(data)

    # Make the data one-hot-encoded because speed is not better than wait
    ohe = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = ohe.fit_transform(integer_encoded)
    return onehot_encoded


def everything(X_train, Y_train, X_val, Y_val, X_test, Y_test, epoch, d1, d2):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint("datasets/feb/results/best_model-{}-{}-{}".format(d1, d2, epoch) +  ".h5", monitor='val_accuracy', 
                        mode='max', verbose=1, save_best_only=True)
    model = Sequential()
    model.add(Dense(d1, input_shape=(7,), activation='relu'))
    model.add(Dense(d2, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.save("datasets/feb/results/{}-{}-{}".format(d1, d2, epoch))
    print("Created model!")

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_val = np.asarray(X_val)
    Y_val = np.asarray(Y_val)

    history = model.fit(
    X_train, 
    Y_train, 
    epochs=epoch, 
    batch_size=64,
    validation_data=(X_val, Y_val),
    callbacks=[es, mc])

    # Save the history
    # https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

    # save to csv:  
    hist_csv_file = 'datasets/feb/results/history-{}-{}-epoch-{}.json'.format(d1, d2, epoch) 
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    
    print("Trained the model!")

    y_pred = model.predict(X_test)

    # Converting predictions to label
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))

    # Converting one hot encoded test label to label
    test = list()
    for i in range(len(Y_test)):
        test.append(np.argmax(Y_test[i]))
    
    print("Result: ", test[4895:4900])
    print("Y_test: ", pred[4895:4900])
    # Get accuracy
    a = accuracy_score(test, pred)
    print("Accuracy is: ", a * 100)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig("datasets/feb/results/Model accuracy-{}-{}-epoch-{}.png".format(d1, d2, epoch))

    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    # plt.show()
    plt.savefig("datasets/feb/results/Model loss-{}-{}-epoch-{}.png".format(d1, d2, epoch))
    plt.clf()
    print("Last acc: ", history.history["accuracy"][-1])

    global best_acc, best_d1, best_d2, best_epoch
    if (a > best_acc):
        best_acc = a
        best_d1 = d1
        best_d2 = d2
        best_epoch = epoch
    print("******** Best d1: ", best_d1, " best d2: ", best_d2, " best epoch: ", best_epoch)


def main():
    train_data = pd.read_csv("datasets/feb/training/training-peers-rich.csv", dtype='category')
    test_data = pd.read_csv("datasets/feb/training/test-peers-rich.csv", dtype='category')
    validation_data = pd.read_csv("datasets/feb/training/validation-peers-rich.csv", dtype='category')
    print("Shape of train: ", train_data.shape)

    main_cols = ['relative_x', 
            'relative_y', 
            'RelVelocity_X', 
            'RelVelocity_Y',
            'AbsVelocity_X',
            'AbsVelocity_Y',
            'EgoHeadingRad'
    ]
    peer_cols = ['PeerX',
            'PeerY',
            'PeerRelVelX',
            'PeerRelVelY',
            'PeerAbsVelX',
            ]

    X_train, Y_train = getNormalizedData(train_data, main_cols, [], False, "datasets/feb/training/train_X_norm.csv")
    Y_train = oneHotEncoder(Y_train)
    X_val, Y_val = getNormalizedData(validation_data, main_cols, [], False, "datasets/feb/training/val_X_norm.csv")
    Y_val = oneHotEncoder(Y_val)
    X_test, Y_test = getNormalizedData(test_data, main_cols, [], False, "datasets/feb/training/test_X_norm.csv")
    Y_test = oneHotEncoder(Y_test)
    print("Done loading and normalizing!")

    # params = {
    #     "epoch": [5, 11, 50, 100, 200],
    #     "d1": [16, 32, 100, 250, 512],
    #     "d2": [12, 16, 70, 150, 275]
    # }

    # for i in range(4):
    #     epoch = params["epoch"][i]
    #     d1 = params["d1"][i]
    #     d2 = params["d2"][i]
    #     everything(X_train, Y_train, X_val, Y_val, X_test, Y_test, epoch, d1, d2)

    everything(X_train, Y_train, X_val, Y_val, X_test, Y_test, 200, 512, 275)
    # print()
    # print("Best d1: ", best_d1)
    # print("Best d2: ", best_d2)
    # print("Best epoch: ", best_epoch)

    # Best d1:  100
    # Best d2:  70
    # Best epoch:  50

    # Now best:
    # Best d1: 250
    # Best d2: 150
    # Best epoch: 100

if __name__ == "__main__":
    main()