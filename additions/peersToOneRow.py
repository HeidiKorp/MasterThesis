import pandas as pd
import numpy as np
import re

def splitFeature(feat):
    feat = getSubstring(feat)
    feat = feat.split(",")

    # feat = [[x] for x in feat]
    # feat = np.array(feat)
    # feat = feat.T
    return feat

def getSubstring(x):
    start = x.find("[") + 1
    end = x.find("]")
    res = re.sub("[\[\]\' ]", "", x[start:end])
    return res

def countPeers(peerX):
    feat = getSubstring(peerX)
    feat = re.sub("[\[\]\' ]", "", feat)
    feat = feat.split(",")
    if len(feat) == 1 and feat[0] == '':
        return 0
    return len(feat)

def applyPeerCount(data):
    data['peerCount'] = data.apply(lambda x: countPeers(x['PeerX']), axis=1)
    return data

def listToCols(data):
    # print("Data 1:\n", data)
    # data = [[x] for x in data]
    # print("Data2:\n", data)
    data = np.array(data)
    # data = data.T
    # print("Data 3:\n", pd.DataFrame(data))
    return data.T

def peersToFeatures(egoX, egoY, egoHeading, egoRelVelX, egoRelVelY, peerX, peerY, peerRelVelX, peerRelVelY):
    peerX_val = getSubstring(peerX)
    peerY_val = getSubstring(peerY)
    peerRelVelX_val = getSubstring(peerRelVelX)
    peerRelVelY_val = getSubstring(peerRelVelY)

    peerX_val = peerX_val.split(',')
    if peerX_val[0] == '':
        peerX_val, peerY_val, peerRelVelX_val, peerRelVelY_val = [], [], [], []
    else:
        peerY_val = peerY_val.split(',')
        peerRelVelX_val = peerRelVelX_val.split(',')
        peerRelVelY_val = peerRelVelY_val.split(',')

    # print("PeerY: \n", peerY_val)

    while len(peerX_val) < 3:
        peerX_val.append(egoX)
        peerY_val.append(egoY)
        peerRelVelX_val.append(0.0)
        peerRelVelY_val.append(0.0)

    vals = [peerX_val, peerY_val, peerRelVelX_val, peerRelVelY_val]
    vals = np.array(vals).T
    vals = vals.tolist()
    res = [egoX, egoY, egoHeading, egoRelVelX, egoRelVelY] + vals[0] + vals[1] + vals[2]

    return pd.Series(res)


def getPeerCols(maxPeers, colNames):
    cols = []
    for i in range(1, maxPeers + 1):
        for j in colNames:
            cols.append(j + "_" + str(i))
    return cols

def calcAction(velX, velY):
    thresh = 7.23
    vel_sum = abs(velX) + abs(velY)
    if vel_sum == 0:
        return 'wait'
    elif vel_sum < thresh:
        return 'slow'
    else:
        return 'speed'

def addAction(data):
    def lambdafunc(row): return calcAction(row['RelVelocity_X'], row['RelVelocity_Y'])
    data['futAction'] = data.apply(lambdafunc, axis=1)
    return data

def filterOutHeader(inFile, outFile):
    data = pd.read_csv(inFile, dtype='category')
    data.drop(data.index[data['RelVelocity_X'] == 'RelVelocity_X'], inplace = True)
    data.to_csv(outFile)


def main():
    # data = pd.read_csv("datasets/feb/slowing-thresh-peers-action.csv")
    # data = applyPeerCount(data)
    # data.to_csv("datasets/feb/slowing-thresh-peers-count.csv")

    # wait_data = pd.read_csv("datasets/feb/waiting-thresh-peers-count.csv")
    # speeding_data = pd.read_csv("datasets/feb/speeding-thresh-peers-count.csv")
    # slowing_data = pd.read_csv("datasets/feb/slowing-thresh-peers-count.csv")

    # max_wait_peers = wait_data['peerCount'].max()
    # max_speed_peers = speeding_data['peerCount'].max()
    # max_slow_peers = speeding_data['peerCount'].max()

    # print("Max peers in wait: ", max_wait_peers)
    # print("Max peers in speed: ", max_speed_peers)
    # print("Max peers in slow: ", max_slow_peers)

    # Max peers is 3


    data = pd.read_csv("datasets/feb/training/test-peers-rich-futAction-no-head.csv")
    # wait_data = wait_data[11138:11148] # 2 peers
    # wait_data = wait_data[:5] # 0 peers
    # wait_data = wait_data[1233:1243] # 1 peer

    def lambdafunc(row): return peersToFeatures(
                                row['relative_x_trans'], row['relative_y_trans'], 
                                row['EgoHeadingRad'], row['RelVelocity_X'], row['RelVelocity_Y'],
                                row['PeerX'], row['PeerY'], 
                                row['PeerRelVelX'], row['PeerRelVelY'])

    res = pd.DataFrame()
    res = data.apply(lambdafunc, axis=1)
    cols = getPeerCols(3, ['PeerX', 'PeerY', 'PeerRelVelX', 'PeerRelVelY'])
    res.columns = ['relative_x_trans', 'relative_y_trans', 'EgoHeadingRad', 'RelVelocity_X', 'RelVelocity_Y'] + cols
    res = addAction(res)

    # res = pd.concat([data[['relative_x_trans', 'relative_y_trans', 'EgoHeadingRad', 'RelVelocity_X', 'RelVelocity_Y']], peers], ignore_index=True)

    res.to_csv("datasets/feb/training/test-peers-split-futAction.csv")

    # Get a dataset where peer and orig data have been combined!


    # Next:
    # Remove uniqueId and peer timestamp from the result (they were added for debugging)
    # Run the script above for every file
    # Save the nr of longest columns (max nr of peers)

    # For all rows where nr of peers is less than that:
        # Add dummy data where x and y are the same as ego vehicle
        # and speed is 0
    


if __name__ == "__main__":
    main()