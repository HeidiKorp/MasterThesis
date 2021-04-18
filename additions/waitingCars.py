from transformTracks import getThreshold, getCenterPoint
from drawTrajectory import plotTrajectory
import pandas as pd
import itertools
import numpy as np

queenWaiting = 512
leithWaiting = 1513
oliverWaiting = 328
orchardWaiting = 567


def saveWaitingTracks(idFile, dataFile, outFile):
    ids = pd.read_csv(idFile, dtype='category').to_numpy()
    print("la")
    data = pd.read_csv(dataFile, dtype='category')
    print("ke")

    res = pd.DataFrame()
    print("Hey")
    counter = 0
    for row in ids:
        csvName = row[-1]
        objId = row[-2]
        print(counter)
        val = data.loc[(data['ObjectId'] == objId) & (data['csv_name'] == csvName)]
        print("Data: ", len(val.index))
        res = res.append(val)
        print("Res: ", len(res.index))
        print()
        counter += 1
    # res = data.loc[(data['ObjectId'] == ids['ObjectId']) & (data['csv_name'] == ids['csv_name'])]
    print(res.columns)
    res.columns = data.columns
    res.to_csv(outFile)


def testPandas():
    data = {'A': [1,4,1,2, 2], 'B': [2,3,4, 4, 4]}
    df = pd.DataFrame(data)
    print(df)
    print()
    res = df.loc[(df['A'] == 1) & (df['B'] == 2)]
    print(res)
    # df2 = df.drop_duplicates()
    # print(df2)


def getDataWithoutWaiting(waitingFile, dataFile, outFile):
    wait = pd.read_csv(waitingFile, dtype='category')
    data = pd.read_csv(dataFile, dtype='category')

    print("Read data files")
    # ids = data[['ObjectId', 'csv_name']]
    # ids = ids.drop_duplicates()
    df_all = data.merge(wait, on=['ObjectId', 'csv_name'], how='left', indicator=True)
    
    print("All head:\n", df_all.head())
    res = df_all.loc[df_all['_merge'] == 'left_only']
    print("Res head:\n", res.head())
    res.to_csv(outFile)


def getNotWaitingSplit(idFile, dataFile, outFile):
    ids = pd.read_csv(idFile, dtype='category')
    data = pd.read_csv(dataFile, dtype='cateogry')




def splitIds(fileName, outName, count):
    data = pd.read_csv(fileName, dtype='category')

    queen = data.loc[data['csv_name'].str.contains("queen")]
    queen = queen[['ObjectId', 'csv_name']]
    queen = queen.drop_duplicates()

    leith = data.loc[data['csv_name'].str.contains("leith")]
    leith = leith[['ObjectId', 'csv_name']]
    leith = leith.drop_duplicates()

    oliver = data.loc[data['csv_name'].str.contains("oliver")]
    oliver = oliver[['ObjectId', 'csv_name']]
    oliver = oliver.drop_duplicates()

    orchard = data.loc[data['csv_name'].str.contains("orchard")]
    orchard = orchard[['ObjectId', 'csv_name']]
    orchard = orchard.drop_duplicates()

    
    res = queen.sample(n = count)
    res = res.append(leith.sample(n = count))
    res = res.append(oliver.sample(n = count))
    res = res.append(orchard.sample(n = count))
    res.to_csv(outName)


def saveWaitingPandasIds(inFile, outFile):
    data = pd.read_csv(inFile, dtype='category')
    relX = data.loc[data['RelVelocity_X'].astype(float) == 0.0]
    relY = data.loc[data['RelVelocity_Y'].astype(float) == 0.0]
    absX = data.loc[data['AbsVelocity_X'].astype(float) == 0.0]
    absY = data.loc[data['AbsVelocity_Y'].astype(float) == 0.0]
    
    chained = relX.append([relY, absY, absX])
    chained_unique = chained[['ObjectId', 'csv_name']].drop_duplicates()
    chained_unique.to_csv(outFile)


def sortDataframe(fileName):
    data = pd.read_csv(fileName, dtype='category')
    data = data.sort_values(['ObjectId', 'Timestamp'])
    data.to_csv(fileName)


def countIntersectionTracks(fileName):

    data = pd.read_csv(fileName, dtype='category')
    queen = data.loc[data['csv_name'].str.contains("queen")]
    leith = data.loc[data['csv_name'].str.contains("leith")]
    oliver = data.loc[data['csv_name'].str.contains("oliver")]
    orchard = data.loc[data['csv_name'].str.contains("orchard")]
    roslyn = data.loc[data['csv_name'].str.contains("roslyn")]

    print("Queen: ", queen.count, "\nLeith: ", leith.count, "\nOliver: ", oliver.count, "\nOrchard: ", orchard.count, "\nRoslyn: ", roslyn.count)

  
def getBeforeThreshold(fileName, outputFile):
    data = pd.read_csv(fileName)
    northThresh, southThresh, eastThresh, westThresh = getThreshold()
    print("North: ", northThresh, "\nWest: ", westThresh, "\nSouth: ", southThresh, "\nEast: ", eastThresh)

    northData = data.loc[(data['origin'] == 'north') & (data['relative_y_trans'].astype(float) > northThresh)]
    westData = data.loc[(data['origin'] == 'west') & (data['relative_x_trans'].astype(float) < westThresh)]
    southData = data.loc[(data['origin'] == 'south') & (data['relative_y_trans'].astype(float) < southThresh)]
    eastData = data.loc[(data['origin'] == 'east') & (data['relative_x_trans'].astype(float) > eastThresh)]
    
    beforeEntry = northData.append([westData, southData, eastData])
    print(beforeEntry.columns)
    # beforeEntry.columns = data.columns
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
    # countIntersectionTracks('waitingIds.csv')
    # getBeforeThreshold("../../intersections-dataset-transformed.csv", "intersections-dataset-before-thresh.csv")
    # data = pd.read_csv(fileName, dtype='category')
    # data = pd.read_csv("../../records/records_0-5000.csv", dtype='category')
    # centX, centY = getCenterPoint()
    # plotTrajectory(data, 'beforeEntry.png', centX, centY, getThreshold())
    # sortDataframe(fileName)
    # getOneTrack('1', fileName, outName)
    # getOneFile("oliver-wyndora_05", fileName, "oliver05.csv")

    # data = pd.read_csv("../../intersections-dataset.csv", dtype='category')
    # print(len(data.ObjectId.unique()))
    # testPandas()
    # saveWaitingPandasIds("intersections-dataset-before-thresh.csv", 'waitingIds.csv')
    # saveWaitingTracks('waitingIds.csv', "intersections-dataset-before-thresh.csv", 'waiting-thresh.csv')
    # getDataWithoutWaiting('datasets/waitingIds.csv', "datasets/intersections-dataset-before-thresh.csv", "datasets/dataset-without-waiting.csv")
    # splitIds("datasets/dataset-without-waiting.csv", "datasets/not-waiting-split-ids.csv", 328)
    saveWaitingTracks("datasets/not-waiting-split-ids.csv", 'datasets/intersections-dataset-before-thresh.csv', 'datasets/not-waiting-thresh-split.csv')

if __name__ == "__main__":
    main()