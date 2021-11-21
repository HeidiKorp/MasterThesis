import pandas as pd
import numpy as np

# Splits the dataset into 60%, 20%, 20% (train, val, test)
def makeTrainValTest(dataFile, outTrain, outValidate, outTest):
    data = pd.read_csv(dataFile, dtype='category')
    train, validate, test = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])
    
    train.to_csv(outTrain, mode='a')
    validate.to_csv(outValidate, mode='a')
    test.to_csv(outTest, mode='a')


def mergePeersToData(dataFile, peersFile, outFile):
    data = pd.read_csv(dataFile, dtype='category')
    peers = pd.read_csv(peersFile, dtype='category')

    print("Done reading!")
    peers = peers.rename(columns={'UniqueId': 'uniqueId'})

    data = data.iloc[:, 1:]
    peers = peers.iloc[:, 2:]

    # print("Data cols: ", data.columns)
    # print("Peer cols: ", peers.columns)
    res = data.merge(peers, how='inner', on=['uniqueId', 'Timestamp'])
    print(res.columns)
    res.to_csv(outFile)

def main():
    # makeTrainValTest("datasets/speeding-thresh-peers.csv", \
    #                 "datasets/training/training-peers.csv", \
    #                 "datasets/training/validation-peers.csv", \
    #                 "datasets/training/test-peers.csv")
    # addPeersToOrigData("datasets/intersections-dataset-before-thresh.csv", \
    #                 "datasets/training/test.csv", \
    #                 "datasets/training/test-rich.csv")

    mergePeersToData("datasets/intersections-dataset-before-thresh.csv", \
                "datasets/training/training-peers.csv", \
                "datasets/training/training-peers-rich.csv")

    mergePeersToData("datasets/intersections-dataset-before-thresh.csv", \
                "datasets/training/validation-peers.csv", \
                "datasets/training/validation-peers-rich.csv")


if __name__ == "__main__":
    main()