from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import random
import math
import pandas as pd

def normalizeFeature(data, feature, newCol):
    le = LabelEncoder()
    data[newCol] = le.fit_transform(data[feature])
    return data
    # return dict(zip(le.classes_, le.transform(le.classes_)))

def unNormalize(mapping, res):
    for dest, code in mapping:
        if code == res:
            return dest

def normalizeData(data):
    data = data.astype(float)
    sc = StandardScaler()
    res = sc.fit_transform(data)
    return pd.DataFrame(res, columns=data.columns)

def split_sequences(data, n_steps, track_id):
    X, y = list(), list()
    for i in range(len(data)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(data):
            break
        elif i == 0 and data[0][track_id] != data[1][track_id]:
            continue
        # If the track_id of current instance is not the same as the one of the previous instance
        elif data[end_ix-1][track_id] != data[end_ix-2][track_id]:
            i = end_ix
        else:
            seq_x, seq_y = data[i:end_ix, :-2], data[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    return X.astype(np.float), y.astype(np.float)


def get_train_val_test(data, train_size, val_size, test_size):
    # Get all unique values of column 'uniqueId'
    unique_tracks = data['uniqueId'].unique()
    nr_tracks = len(unique_tracks)

    # Get train, val, test size in tracks
    train_z = round(nr_tracks * train_size)
    val_z = round(nr_tracks * val_size) + train_z
    test_z = round(nr_tracks * test_size) + val_z
    # Shuffle tracks
    random.shuffle(unique_tracks)
    # Get row indexes where uniqueId equals the n first track numbers
    # n equals the number of train tracks
    train_idx = data.index[data['uniqueId'].isin(unique_tracks[:train_z])].tolist()
    train = data.iloc[train_idx]

    val_idx = data.index[data['uniqueId'].isin(unique_tracks[train_z:val_z])].tolist()
    val = data.iloc[val_idx]

    test_idx = data.index[data['uniqueId'].isin(unique_tracks[val_z:])].tolist()
    test = data.iloc[test_idx]
    return train, val, test