import pandas as pd
import numpy as np
from ast import literal_eval

# Here the timestamp is wrong!!!


def getOtherVehicles(timestamp, uniqueId, data):
    # print("Got here!")
    new_timestamp = float(timestamp)
    data['Timestamp'] = data['Timestamp'].astype(float)
    data['time_diff'] = abs(data['Timestamp'] - new_timestamp)

    sub = data.loc[(data['time_diff'] < 1) & ~((data['uniqueId'] == uniqueId))
                   & ~((data['AbsVelocity_X'] == '0.0') | (data['AbsVelocity_Y'] == '0.0'))]
    # print("Where time diff < 1: ", len(sub.index))
    sub = sub.sort_values(by='time_diff', ascending=True)
    sub = sub.drop_duplicates(subset=['uniqueId'], keep='first')

    peerX = sub['relative_x_trans'].to_list()
    peerY = sub['relative_y_trans'].to_list()
    peerRelVelX = sub['RelVelocity_X'].to_list()
    peerRelVelY = sub['RelVelocity_Y'].to_list()
    peerAbsVelX = sub['AbsVelocity_X'].to_list()
    peerAbsVelY = sub['AbsVelocity_Y'].to_list()
    peerUniqueId = sub['uniqueId'].to_list()
    # peerObjectId = sub['ObjectId'].to_list()
    # peerCsvName = sub['csv_name'].to_list()
    peerTimestamp = sub['time_diff'].to_list()
    # timestamp = str(timestamp)
    print("Timestamp: ", timestamp, " peerTime: ", peerTimestamp)

    return pd.Series([
        # objId,
        # csvName,
        uniqueId,
        timestamp,
        peerX,
        peerY,
        peerRelVelX,
        peerRelVelY,
        peerAbsVelX,
        peerAbsVelY,
        peerUniqueId,
        # peerObjectId,
        # peerCsvName,
        peerTimestamp
    ])


def removePeersWhoAreWaitingToo(dataFile):
    data = pd.read_csv(dataFile, dtype='category')
    data = data.loc[~((data['PeerAbsVelX'] == '0.0') | (data['PeerAbsVelY'] == '0.0'))]
    data.to_csv(dataFile)


def applyOtherVehicles(inFile, dataFile, outFile):
    filtered = pd.read_csv(inFile, dtype='category')
    data = pd.read_csv(dataFile, dtype='category')
    # data['Timestamp'] = data['Timestamp'].astype(float)

    # # Get only rows where velocity is 0
    # relX = filtered.loc[filtered['RelVelocity_X'].astype(float) == 0.0]
    # relY = filtered.loc[filtered['RelVelocity_Y'].astype(float) == 0.0]
    # absX = filtered.loc[filtered['AbsVelocity_X'].astype(float) == 0.0]
    # absY = filtered.loc[filtered['AbsVelocity_Y'].astype(float) == 0.0]

    # chained = relX.append([relY, absY, absX])
    # chained_unique = chained.drop_duplicates()

    print("Starting filtering!")
    # print("Velocity 0 length: ", len(chained_unique.index))
    def lambdafunc(row): return getOtherVehicles(
        row['Timestamp'], row['uniqueId'], data)

    # chained_unique[['Peer_X', 'Peer_Y', 'PeerRelVel_X', 'PeerRelVel_Y', 'PeerAbsVel_X', 'PeerAbsVel_Y', 'PeerObjectId', 'PeerCsvName']] =\
    res = pd.DataFrame()
    res = \
        filtered.apply(lambdafunc, axis=1)
    print("Columns: ", res.columns)
    res.columns = ['UniqueId', 'Timestamp', 'PeerX', 'PeerY', 'PeerRelVelX', 'PeerRelVelY', 'PeerAbsVelX', 'PeerAbsVelY', 'PeerUniqueId', 'PeerTimestamp']
    print("Done here!")
    # print(res.head())
    res.to_csv(outFile)


def test():
    df = pd.DataFrame({'B': [[1, 2], [1, 2]]})
    print(df)
    print()
    df = df.explode('B')
    print(df)


def setColumns(inFile, columns):
    data = pd.read_csv(inFile, dtype='category')
    data.columns = columns
    data.to_csv(inFile)


def printTimestampDifference(inFile):
    data = pd.read_csv(inFile, dtype='category')
    sub = data.loc[(data['ObjectId'] == '146') & (
        data['csv_name'] == 'split_20180116-082129-urban-stationary-queen-hanks_17.csv')]
    timestamps = sub['Timestamp'].to_numpy()
    timestamps = [float(x) for x in timestamps]
    # timestamps = [int(x) for x in timestamps]

    diffs = []

    for i in range(len(timestamps) - 1):
        diff = timestamps[i + 1] - timestamps[i]
        print(diff)
        if diff < 1:
            diffs.append(diff)
    aver = sum(diffs) / len(diffs)
    print("\nAverage is: ", aver)
    print("\nHalf average is: ", aver / 2)


def getNotEmptyTracks(inFile, outFile):
    data = pd.read_csv(inFile, dtype='category')
    sub = data.loc[~(data.Peer_X == '[]')]
    # ids = sub[['ObjectId', 'csv_name']].drop_duplicates()
    # res = data.merge(ids, on=['ObjectId', 'csv_name'], how='inner')
    sub.to_csv(outFile)


def concatDatasets(dataFile, peersFile, outFile):
    data = pd.read_csv(dataFile, dtype='category')
    peers = pd.read_csv(peersFile, dtype='category')
    res = data.merge(
        peers, on=['ObjectId', 'csv_name', 'Timestamp'], how='inner')
    res.to_csv(outFile)


def getPeersPercentage(dataFile):
    data = pd.read_csv(dataFile, dtype='category')
    dataPeers = data.loc[data.PeerX != '[]']
    perc = round(len(dataPeers) / len(data) * 100, 2)
    print("Percentage is: ", perc)


def addActionCategory(dataFile, action, outFile):
    data = pd.read_csv(dataFile, dtype='category')
    data['action'] = action
    data.to_csv(outFile)


# def getPeersPercentage(peersFile):
#     data = pd.read_csv(peersFile, dtype='category')
#             objId,
#         csvName,
#         timestamp,
#         peerX,
#         peerY,
#         peerRelVelX,
#         peerRelVelY,
#         peerAbsVelX,
#         peerAbsVelY,
#         peerObjectId,
#         peerCsvName,
#         peerTimestamp
#     rowsWithPeers = data['']


def main():
    # applyOtherVehicles('datasets/waiting-before-thresh.csv', "datasets/intersections-dataset-before-thresh.csv", "datasets/waiting-thresh-peers.csv")
    # print("Done waiting")
    # applyOtherVehicles('datasets/slowing-before-thresh.csv',
    #                    "datasets/intersections-dataset-before-thresh.csv", "datasets/slowing-thresh-peers.csv")
    # print("Done slowing")
    # applyOtherVehicles('datasets/speeding-before-thresh.csv',
    #                    "datasets/intersections-dataset-before-thresh.csv", "datasets/speeding-thresh-peers.csv")
    # print("Done speeding")
    # removePeersWhoAreWaitingToo("datasets/speeding-thresh-peers.csv")
    # printTimestampDifference('datasets/waiting-thresh-split.csv')
    # setColumns("datasets/not-waiting-thresh-peers.csv", ['id', 'ObjectId', 'csv_name', 'Timestamp', 'Peer_X', 'Peer_Y', 'PeerRelVel_X', 'PeerRelVel_Y', 'PeerAbsVel_X', 'PeerAbsVel_Y', 'PeerObjectId', 'PeerCsvName', 'PeerTimeDiff'])
    # updatePeers("datasets/peers3.csv", 'datasets/peers4.csv')
    # test()
    # getNotEmptyTracks("datasets/not-waiting-rich.csv", "datasets/not-waiting-rich-not-empty.csv")
    # concatDatasets("datasets/intersections-dataset-before-thresh.csv", "datasets/not-waiting-thresh-peers.csv", "datasets/not-waiting-rich.csv")
    # getPeersPercentage("datasets/slowing-thresh-peers.csv")
    addActionCategory("datasets/speeding-thresh-peers.csv", "speed", "datasets/speeding-thresh-peers2.csv")


if __name__ == "__main__":
    main()
