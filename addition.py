import pandas as pd
import numpy as np
from helper import getMultiDict

def getRoslynPercentage(fileIn):
    data = pd.read_csv(fileIn, dtype='category')
    sub = data.loc[data.csv_name.str.contains('roslyn')]
    print("Roslyn percentage: ", len(sub) / len(data))

def getNetLenInSeconds(data, netLen):
    sub = data.loc[data.uniqueId == '3']

    s1 = sub[:netLen]
    s2 = sub[netLen:netLen*2]


    t1 = float(s1.iloc[-1].Timestamp) - float(s1.iloc[0].Timestamp)
    t2 = float(s2.iloc[-1].Timestamp) - float(s2.iloc[0].Timestamp)
    print("Len is: ", netLen)
    print("Time 1: ", t1, " time 2: ", t2)
    print()


def removeStrRows(fileIn, col, outFile):
    data = pd.read_csv(fileIn, dtype='category')
    print(data.head())
    data.drop(data.index[data[col] == col], inplace=True)
    data.to_csv(outFile)


def reshapeTest():
    a = list()
    s = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]]
    s = np.asarray(s)
    res = s.reshape((2, 2, s.shape[1]))
    print(res)
    print(res.shape)
    print()
    a.append(res)
    a.append(res)
    print(a)
    print(type(a))
    a = np.array(a)
    print(a)
    print(a.shape)


def saveTestDataClasses(dataFile, testFile, netLen):
    data = pd.read_csv(dataFile, dtype='category')
    test = pd.read_csv(testFile, dtype='category')
    # classes = ['leith']
    classes = ['leith', 'roslyn', 'oliver', 'orchard', 'queen']
    for i in classes:
        sub = data.loc[data.csv_name.str.contains(i)]
        ids = sub.uniqueId.unique().tolist()
        test_sub = test[np.isin(test['uniqueId'].to_numpy(), ids)]        
        test_sub.to_csv("additions/datasets/april/testFiles/test_" + i + "_" + netLen +  ".csv")

def getRoundaboutPercentage(fileIn):
    data = pd.read_csv(fileIn, dtype='category')
    dataLen = len(data)
    classes = ['leith', 'roslyn', 'oliver', 'orchard', 'queen']
    for i in classes:
        sub = data.loc[data.csv_name.str.contains(i)]
        perc = len(sub) / dataLen * 100
        print("Class ", i, " percentage is: ", str(perc))


def saveTestSvmClasses(dataFile, testFile):
    data = pd.read_csv(dataFile, dtype='category')
    test = pd.read_csv(testFile, dtype='category')
    classes = ['leith', 'roslyn', 'oliver', 'orchard', 'queen']
    for i in classes:
        sub = data.loc[data.csv_name.str.contains(i)]

        # res = pd.concat([test, sub], join='left', keys=['relative_x_trans', 'relative_y_trans'])
        test_sub = test.merge(sub, on=['relative_x_trans', 'relative_y_trans'], suffixes=(None, '_y'))
        # relX = sub.relative_x_trans.tolist()
        # relY = sub.relative_y_trans.tolist()
        # rels = zip(relX, relY)
        # test_sub = test[np.isin(zip(test.relative_x_trans.tolist(), test.relative_y_trans.tolist()), rels)]
        print("Res cols: ", test_sub.columns)
        cols = test.columns

        cols = [x for x in cols if 'Unnamed' not in x]
        test_sub = test_sub[cols]
        print("Test sub: ", test_sub.head())
        test_sub.to_csv("additions/datasets/april/testFiles/SVMtest_" + i + ".csv")


def getCommonIds(waitFile, speedFile, slowFile):
    wait_data = pd.read_csv(waitFile, dtype='category')
    speed_data = pd.read_csv(speedFile, dtype='category')
    slow_data = pd.read_csv(slowFile, dtype='category')

    wait_ids = wait_data.UniqueId.unique()
    speed_ids = speed_data.UniqueId.unique()
    slow_ids = slow_data.UniqueId.unique()

    wait_ids = set(wait_ids)
    intersect_1 = wait_ids.intersection(speed_ids)
    print("Type: ", type(intersect_1))

    intersect_2 = intersect_1.intersection(slow_ids)
    print("Common ids:\n", list(intersect_2))



def splitDatasetRoundaboutPosition(fileIn, firstExit, secondExit, thirdExit):
    data = pd.read_csv(fileIn, dtype='category')

    firstEx = {"south": "west", 
                "west": "north", 
                "north": "east", 
                "east": "south",
                "NE": "SE",
                "SE": "SW",
                "SW": "NW",
                "NW": "NE"}

    secondEx = {"south": "north",
                "west": "east",
                "north": "south",
                "east": "west",
                "NE": "SW",
                "SE": "NW",
                "SW": "NE",
                "NE": "SE"}

    thirdEx = {"south": "east",
                "west": "south",
                "north": "west",
                "east": "north",
                "NE": "NW",
                "SE": "NE",
                "SW": "SE",
                "NE": "SW"}
    exits = [firstEx, secondEx, thirdEx]
    exitFiles = [firstExit, secondExit, thirdExit]

    for ex, exFile in zip(exits, exitFiles):
        res = pd.DataFrame()
        for key, val in ex.items():
            sub = data.loc[(data.origin == key) & (data.destination == val)]
            sub = sub.uniqueId
            sub.to_csv(exFile, mode='a')
            print("Finished {} " % ex)


def getExitTestData(testFile, exitFile, outFile):
    test_data = pd.read_csv(testFile, dtype='category')
    exit_data = pd.read_csv(exitFile, dtype='category')
    
    test_ids = test_data.uniqueId.unique()
    exit_ids = exit_data.uniqueId.unique()

    inter = set(test_ids).intersection(exit_ids)
    inter = list(inter)

    res = test_data[np.isin(test_data.uniqueId.to_numpy(), inter)]
    res = res[res.columns.drop(list(res.filter(regex='Unnamed')))]
    res.to_csv(outFile)

def analyzeEvals(dataFile, net_len):
    correct_preds = []
    false_preds = []
    counter = 0
    net_time = 0.2

    if net_len == 10:
        net_time = 0.4
    elif net_len == 15:
        net_time = 0.6

    with open(dataFile) as file:
        for line in file:
            counter += 1
            data = line.split(",")
            # data = np.array(data)
            if data[2] != data[3]:
                correct_preds.append(data)
            else:
                false_preds.append(data)
    nr_correct_preds = len(correct_preds)
    print("Nr correct preds: ", nr_correct_preds, " perc: ", (nr_correct_preds/counter) * 100, " nr examples: ", counter)

    last_false = [el[2] for el in correct_preds]
    # print(last_false)
    last_false = [int(x) for x in last_false]
    last_false = np.where(last_false == -1, 0, last_false)
    avg_false = np.average(last_false)
    print("Average false: ", avg_false, " avg converge time: ", avg_false * net_time)


def main():
    # fileIn = "additions/datasets/intersections-dataset-transformed-1000.csv"
    # data = pd.read_csv(fileIn, dtype='category')
    # getNetLenInSeconds(data, 5)
    # getNetLenInSeconds(data, 15)
    # getNetLenInSeconds(data, 20)
    # getRoslynPercentage("additions/datasets/feb/intersections-dataset-transformed.csv")
    # getMultiDict("additions/datasets/april/ids-to-use-no-text.txt", "additions/datasets/april/ids-dict.txt")
    # removeStrRows("additions/datasets/april/intersections-dataset-transformed-balanced-duplicate.csv", 'code', "additions/datasets/april/intersections-dataset-transformed-balanced-duplicate-clean.csv")
    # reshapeTest()
    # train = pd.read_csv("additions/datasets/april/train-" + str(5) + ".csv", dtype='category')
    # val = pd.read_csv("additions/datasets/april/validation-" + str(5) + ".csv", dtype='category')
    # test = pd.read_csv("additions/datasets/april/test-" + str(5) + ".csv", dtype='category')

    # train = train[train.columns.drop(list(train.filter(regex='Unnamed')))]
    # val = val[val.columns.drop(list(val.filter(regex='Unnamed')))]
    # test = test[test.columns.drop(list(test.filter(regex='Unnamed')))]

    # train.to_csv("additions/datasets/april/train-" + str(5) + "2.csv")
    # val.to_csv("additions/datasets/april/validation-" + str(5) + "2.csv")
    # test.to_csv("additions/datasets/april/test-" + str(5) + "2.csv")

    # addCsvNameShort("additions/datasets/feb/intersections-dataset-transformed.csv", "additions/datasets/april/intersections-dataset-transformed-csv.csv")
    # saveTestDataClasses("additions/datasets/feb/intersections-dataset-transformed.csv", "additions/datasets/april/test-15.csv", '15')
    # saveTestSvmClasses("additions/datasets/feb/intersections-dataset-transformed.csv", "additions/datasets/feb/training/test-peers-split.csv")
    # getRoundaboutPercentage("additions/datasets/feb/intersections-dataset-transformed.csv")
    # getCommonIds("additions/datasets/waiting-thresh-peers.csv", "additions/datasets/speeding-thresh-peers.csv", "additions/datasets/slowing-thresh-peers.csv")
    # splitDatasetRoundaboutPosition("additions/datasets/feb/intersections-dataset-transformed.csv", 
    #                         "additions/datasets/april/exitFiles/firstExit.csv", 
    #                         "additions/datasets/april/exitFiles/secondExit.csv", 
    #                         "additions/datasets/april/exitFiles/thirdExit.csv")
    # getExitTestData("additions/datasets/april/test-15.csv",  "additions/datasets/april/exitFiles/thirdExit.csv",  "additions/datasets/april/exitFiles/test-15-exit-3.csv")
    analyzeEvals("additions/datasets/april/exitFiles/test-15-exit-3-res.txt", 15)

if __name__ == "__main__":
    main()