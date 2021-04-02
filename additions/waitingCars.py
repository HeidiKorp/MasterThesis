from transformTracks import getThreshold, getCenterPoint
from drawTrajectory import plotTrajectory
import pandas as pd
import itertools

queenWaiting = 254
leithWaiting = 254
oliverWaiting = 254
orchardWaiting = 254


def saveWaitingTracks(inFile, outFile):
    ids = []
    counter = 0

    with open(inFile) as infile:
        for line in infile:
            s = line.split(",")
            # for i in range(len(s)):
            #     print("i: ", i, " val: ", s[i])
            # break
            # if s[4] == "7" and (s[25] == "0.0" or s[25] == "0.0" or s[29] == "0.0" or s[30] == "0.0"):
            #     print(s)
            if s[5] not in ids and (s[26] == "0.0" or s[27] == "0.0" or s[30] == "0.0" or s[31] == "0.0"):
                ids.append(s[4])
    #             # print()
    #             # print("Yeah ", counter)
    #             # counter += 1
    print("Ids: ", len(ids), " set: ", len(set(ids)))
    ids.sort()
    print(ids)
    # orig = pd.read_csv(fileName, dtype='category')
    # out = orig.loc[orig['ObjectId'].isin(ids)]
    # out.columns = orig.columns
    # out.to_csv(outFile)


def saveWaitingPandas(inFile, outFile):
    data = pd.read_csv(inFile, dtype='category')
    relX = data.loc[data['RelVelocity_X'].astype(float) == 0.0].ObjectId.unique()
    relY = data.loc[data['RelVelocity_Y'].astype(float) == 0.0].ObjectId.unique()
    absX = data.loc[data['AbsVelocity_X'].astype(float) == 0.0].ObjectId.unique()
    absY = data.loc[data['AbsVelocity_Y'].astype(float) == 0.0].ObjectId.unique()
    
    chained = itertools.chain(relX, relY, absX, absY)
    ids_set = set(chained)
    ids_set = [int(x) for x in ids_set]
    ids = list(ids_set) 
    ids.sort()
    print("Ids: ", len(ids), "\n", ids)


def sortDataframe(fileName):
    data = pd.read_csv(fileName, dtype='category')
    data = data.sort_values(['ObjectId', 'Timestamp'])
    data.to_csv(fileName)


def countIntersectionTracks(fileName):
    queen = 0
    leith = 0
    oliver = 0
    orchard = 0
    roslyn = 0

    leithTr = []
    oliverTr = []
    queenTr = []
    orchTr = []
    roslynTr = []
    with open(fileName) as infile:
        for line in infile:
            s = line.split(",")
            inter = s[47]
            track = s[4]
            if "queen" in inter and track not in queenTr: 
                queen += 1
                queenTr.append(track)
            elif "leith" in inter and track not in leithTr: 
                leith += 1
                leithTr.append(track)
            elif "oliver" in inter and track not in oliverTr: 
                oliver += 1
                oliverTr.append(track)
            elif "orchard" in inter and track not in orchTr: 
                orchard += 1
                orchTr.append(track)
            elif "roslyn" in inter and track not in roslynTr: 
                roslyn + 1
                roslynTr.append(track)
    print("Queen: ", queen, "\nLeith: ", leith, 
            "\nOliver: ", oliver, "\nOrchard: ", orchard,
            "\nRoslyn: ", roslyn)


def getBeforeThreshold(fileName, outputFile):
    data = pd.read_csv(fileName)
    northThresh, southThresh, eastThresh, westThresh = getThreshold()
    print("North: ", northThresh, "\nWest: ", westThresh, "\nSouth: ", southThresh, "\nEast: ", eastThresh)

    northData = data.loc[(data['origin'] == 'north') & (data['relative_y'].astype(float) > northThresh)]
    westData = data.loc[(data['origin'] == 'west') & (data['relative_x'].astype(float) < westThresh)]
    southData = data.loc[(data['origin'] == 'south') & (data['relative_y'].astype(float) < southThresh)]
    eastData = data.loc[(data['origin'] == 'east') & (data['relative_x'].astype(float) > eastThresh)]
    
    beforeEntry = northData.append([westData, southData, eastData])
    beforeEntry.columns = data.columns
    beforeEntry.to_csv(outputFile)


def getOneTrack(trackName, inFile, outFile):
    data = pd.read_csv(inFile, dtype='category')
    track = data.loc[(data['ObjectId'] == trackName) & (data['csv_name'].str.contains("oliver-wyndora_05"))]
    track.columns = data.columns
    track.to_csv(outFile)


def getOneFile(trackFile, inFile, outFile):
    data = pd.read_csv(inFile, dtype='category')
    track = data.loc[data['csv_name'].str.contains(trackFile)]
    track.columns = data.columns
    track.to_csv(outFile)



def main():
    # fileName = "waiting-dataset.csv"
    # fileName = "../../records/records_0-5000.csv"
    # fileName = "../../intersections-dataset.csv"
    # fileName = "1stTrackOliver05.csv"
    # fileName = "beforeEntry.csv"
    # outName = "beforeEntry.csv"
    # outName = "waiting-dataset.csv"
    # outName = "1stTrackOliver05.csv"
    # outName = "1srTrackOliver05BeforeEntry.csv"
    # saveWaitingTracks(fileName, outName)
    # saveWaitingPandas(fileName, outName)
    # countIntersectionTracks(fileName, outName)
    getBeforeThreshold("oliver05Transform.csv", "oliver05Before.csv")
    # data = pd.read_csv(fileName, dtype='category')
    # data = pd.read_csv("../../records/records_0-5000.csv", dtype='category')
    # centX, centY = getCenterPoint()
    # plotTrajectory(data, 'beforeEntry.png', centX, centY, getThreshold())
    # sortDataframe(fileName)
    # getOneTrack('1', fileName, outName)
    # getOneFile("oliver-wyndora_05", fileName, "oliver05.csv")

    # data = pd.read_csv("../../intersections-dataset.csv", dtype='category')
    # print(len(data.ObjectId.unique()))



if __name__ == "__main__":
    main()