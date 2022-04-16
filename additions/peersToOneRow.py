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

def peersToFeatures(peerX, peerY, peerRelVelX, peerRelVelY):
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

    print("PeerY: \n", peerY_val)


    return None
    # # print("PeerX: ", peerX)
    # sub = getSubstring(peerX)
    # print("Sub: ", sub)
    # if sub:
    #     if ',' in sub:
    #     print("*** Got here!")
    #     peerX_val, peerX_name = splitFeature(peerX, 'peerX')
    #     peerY_val, peerY_name = splitFeature(peerY, 'peerY')
    #     relVelX_val, relVelX_name = splitFeature(peerRelVelX, 'peerRelVelX')
    #     peerRelVelY_val, peerRelVelY_name = splitFeature(peerRelVelY, 'peerRelVelY')
    #     # peerUniqueId_val, peerUniqueId_name = splitFeature(peerUniqueId, 'peerUniqueId')
    #     # peerTimestamp_val, peerTimestamp_name = splitFeature(peerTimestamp, 'peerTimestamp')
    
    # # cols = np.array([peerX_name, peerY_name, relVelX_name,
    # #         peerRelVelY_name]).flatten()
    
    #     vals = pd.Series([
    #         peerX_val,
    #         peerY_val,
    #         relVelX_val,
    #         peerRelVelY_val
    #     ])
    #     return vals
    # else: # Change it with dummy data!
    #     return None 


def getPeerCols(maxPeers, colNames):
    cols = []
    for i in range(1, maxPeers + 1):
        for j in colNames:
            cols.append(j + "_" + str(i))
    return cols





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


    wait_data = pd.read_csv("datasets/feb/waiting-thresh-peers-count.csv")
    # wait_data = wait_data[11138:11148] # 2 peers
    wait_data = wait_data[:5] # 0 peers
    # wait_data = wait_data[1233:1243] # 1 peer

    def lambdafunc(row): return peersToFeatures(row['PeerX'], row['PeerY'], 
                                row['PeerRelVelX'], row['PeerRelVelY'])

    res = pd.DataFrame()
    res = wait_data.apply(lambdafunc, axis=1)
    # print("Res: \n", res)
    res.columns = getPeerCols(3, ['PeerX', 'PeerY', 'PeerRelVelX', 'PeerRelVelY'])
    # print("Columns: \n", res.columns)



    # Next:
    # Remove uniqueId and peer timestamp from the result (they were added for debugging)
    # Run the script above for every file
    # Save the nr of longest columns (max nr of peers)

    # For all rows where nr of peers is less than that:
        # Add dummy data where x and y are the same as ego vehicle
        # and speed is 0
    


if __name__ == "__main__":
    main()