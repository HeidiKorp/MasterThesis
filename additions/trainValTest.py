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

    res = data.merge(peers, how='inner', on=['uniqueId', 'Timestamp'])
    print(res.columns)
    res.to_csv(outFile)


def getAllPeers(files, outFile):
    for file in files:
        data = pd.read_csv(file, dtype='category')
        data.to_csv(outFile, mode='a')


def main():
    # makeTrainValTest("datasets/feb/speeding-thresh-peers-action.csv", \
    #                 "datasets/feb/training/training-peers.csv", \
    #                 "datasets/feb/training/validation-peers.csv", \
    #                 "datasets/feb/training/test-peers.csv")
    # makeTrainValTest("datasets/feb/slowing-thresh-peers-action.csv", \
    #                 "datasets/feb/training/training-peers.csv", \
    #                 "datasets/feb/training/validation-peers.csv", \
    #                 "datasets/feb/training/test-peers.csv")
    # makeTrainValTest("datasets/feb/waiting-thresh-peers-action.csv", \
    #                 "datasets/feb/training/training-peers.csv", \
    #                 "datasets/feb/training/validation-peers.csv", \
    #                 "datasets/feb/training/test-peers.csv")
    # addPeersToOrigData("datasets/intersections-dataset-before-thresh.csv", \
    #                 "datasets/training/test.csv", \
    #                 "datasets/training/test-rich.csv")

    # mergePeersToData("datasets/feb/intersections-dataset-before.csv", \
    #             "datasets/feb/training/training-peers.csv", \
    #             "datasets/feb/training/training-peers-rich.csv")

    # mergePeersToData("datasets/feb/intersections-dataset-before.csv", \
    #             "datasets/feb/training/test-peers.csv", \
    #             "datasets/feb/training/test-peers-rich.csv")

    mergePeersToData("datasets/feb/intersections-dataset-before.csv", \
                "datasets/feb/training/validation-peers.csv", \
                "datasets/feb/training/validation-peers-rich.csv")


if __name__ == "__main__":
    main()