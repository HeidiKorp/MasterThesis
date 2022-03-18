from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import random
import math
import pandas as pd
import random


le = LabelEncoder()

def normalizeFeature(data, feature, newCol):
    data[newCol] = le.fit_transform(data[feature])
    # print(dict(zip(le.classes_, le.transform(le.classes_))))
    return data
    # return dict(zip(le.classes_, le.transform(le.classes_)))

def unNormalizeFeature(dataCol):
    return le.inverse_transform(dataCol)
# def unNormalize(mapping, res):
#     for dest, code in mapping:
#         if code == res:
#             return dest

def normalizeData(data):
    data = data.astype(float)
    sc = StandardScaler()
    res = sc.fit_transform(data)
    return pd.DataFrame(res, columns=data.columns)


def balancedClasses(data, classCol):
    classes = data[classCol].unique()
    print("CLasses: ", classes)
    minSamples = -1
    # Find the class with least samples
    for c in classes:
        sub = data.loc[data[classCol] == c]
        if len(sub) < minSamples or minSamples < 0:
            minSamples = len(sub)
    print("Min samples is: ", minSamples)

    newData = pd.DataFrame()
    # Get same amount of data for each class
    for c in classes:
        sub = data.loc[data[classCol] == c]
        if len(sub) > minSamples:
            sub2 = pd.DataFrame()
            ids = sub.uniqueId.unique()
            while len(sub2) <= minSamples:
                sub2 = sub2.append(data.loc[data.uniqueId == random.choice(ids)])
            newData = newData.append(sub2)
            print("New data is: ", newData)
        else:
            newData = newData.append(sub)
    print("New data len: ", len(newData))
    return newData

def balanceDupHack(data, classCol, outFile):
    out = open(outFile, 'a')
    classes = [7, 2, 0, 1, 3]
    print("Id type: ", type(data.uniqueId.unique()[0]))
    avgSam = 474420
    currentSum = 0
    ids_to_use = []
    for c in classes:
        sub = data.loc[data[classCol] == c]
        ids = sub.uniqueId.unique()
        used_ids = []
        if len(sub) < avgSam:
            ids_to_use = list(sub.uniqueId.unique())
            currentSum = len(sub)
        while currentSum <= avgSam:
            sel_id = random.choice(ids)
            # while sel_id in used_ids:
            #     sel_id = random.choice(ids)
            used_ids.append(sel_id)
            ids_to_use.append(sel_id)
            currentSum += len(data.loc[data.uniqueId == sel_id])

        listToStr = ' '.join([str(elem) for elem in ids_to_use])
        allIds = ' '.join([str(elem) for elem in ids])
        out.write("Ids to use for class " + str(c) + " are: " + listToStr + "\n\n")
        out.write("All ids for class " + str(c) + " are: " + allIds + "\n\n")
        currentSum = 0
        ids_to_use = []
    out.close()


def writeIdData(data, mul, outFile):
    # print("Data is: ", data.head())
    # newdf = pd.DataFrame(np.repeat(data.values, mul, axis=0))
    newdf = pd.concat([data]*mul).sort_index()
    print("New data is: ", newdf.head())
    newdf.columns = data.columns
    newdf.to_csv(outFile, mode='a')


def getDataById(data, idsFile, outFile):
    fil = open(idsFile, 'r')
    string = fil.read()
    ids = string.split(" ")
    print("First id is: ", ids[0])
    d = {x:ids.count(x) for x in ids}
    print("Nr of keys: ", len(d))

    for key, val in d.items():
        writeIdData(data.loc[data.uniqueId == key], val, outFile)
        print("Wrote id: ", key, " ", str(val), " times")


def countClassLength(fileIn):
    data = pd.read_csv(fileIn, dtype='category')
    classes = data.code.unique()
    for cl in classes:
        sub = data.loc[data.code == cl]
        print("Length for class " + str(cl) + " is: " + str(len(sub)))
    
def balanceDuplicating(data, classCol, outFile, cols):
    classes = data[classCol].unique()
    print("CLasses: ", classes)
    maxSamples = -1
    minSamples = -1
    # Find the class with least samples
    for c in classes:
        sub = data.loc[data[classCol] == c]
        if len(sub) > maxSamples or maxSamples < 0:
            maxSamples = len(sub)
        if len(sub) < minSamples or minSamples < 0:
            minSamples = len(sub)
    avgSam = (minSamples + maxSamples) // 2
    print("Max samples is: ", maxSamples)
    print("Min samples is: ", minSamples)
    print("Avg samples is: ", avgSam)

    # newData = pd.DataFrame()
    # newData = data.copy()
    # Get same amount of data for each class
    for c in classes:
        sub = data.loc[data[classCol] == c]
        ids = sub.uniqueId.unique()
        used_ids = []
        sub2 = pd.DataFrame()
        if len(sub) < avgSam:
            sub2 = sub.copy()
        while len(sub2) <= avgSam:
            sel_id = random.choice(ids)
            while sel_id in used_ids:
                sel_id = random.choice(ids)
            used_ids.append(sel_id)
            sub2 = sub2.append(data.loc[data.uniqueId == sel_id])
        sub2.columns = cols
        sub2.to_csv(outFile, mode='a')
        print("Saved data: ", len(sub2))
    # print("New data len: ", len(newData))
    # return newData

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
            # seq_x, seq_y = data[i:end_ix, :-2], data[end_ix-1, -1]
            seq_x, seq_y = data[i:end_ix, :-2], data[i:end_ix, -1]
            seq_y = [[x] for x in seq_y]
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


def delFirstRows(dataFile, outFile):
    df = pd.read_csv(dataFile, skiprows=789)
    print(df.head())
    df.to_csv(outFile, index=False)

def addTwoFiles(file1, file2, cols, outFile):
    data1 = pd.read_csv(file1, dtype='category')
    data2 = pd.read_csv(file2, dtype='category')

    # Remove rows where code VALUE is code (weird error, weird solution)
    data1 = data1[~data1['code'].isin(['code'])]
    data2 = data2[~data2['code'].isin(['code'])]

    print("Data1 code: ", data1.code.unique())
    print("Data 2 code: ", data2.code.unique())

    res = pd.concat([data2[cols], data1[cols]], axis=0, ignore_index=True)
    res.to_csv(outFile)

# delFirstRows("additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate3.csv", "additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate3-clean.csv")
# countClassLength("additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate.csv")
# addTwoFiles("additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate.csv", 
#             "additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate3-clean.csv", 
#             ['relative_x', 'relative_y', 'EgoHeadingRad', 'AbsVelocity', 'uniqueId', 'code'],
#             "additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate-avg.csv")
# data = pd.read_csv("additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate-avg.csv", dtype='category')
# classes = data.code.unique()
# print(classes)