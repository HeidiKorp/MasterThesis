# from transformTracks import getThreshold, getCenterPoint
from drawTrajectory import plotTrajectory
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt

queenWaiting = 512
leithWaiting = 1513
oliverWaiting = 328
orchardWaiting = 567


def saveWaitingTracks(idFile, dataFile, outFile):
    ids = pd.read_csv(idFile, dtype='category')
    data = pd.read_csv(dataFile, dtype='category')

    res = data.merge(ids, on=['ObjectId', 'csv_name'], how='inner')
    print(res['Timestamp'])
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


def visualizeAvgVelocity(fileIn, visualFile):
    data = pd.read_csv(fileIn, dtype='category')
    data = data.loc[data.csv_name.str.contains("leith")]
    print("Read data!")
    data['AbsVelocity_X'] = abs(data['AbsVelocity_X'].astype(float))
    data['AbsVelocity_Y'] = abs(data['AbsVelocity_X'].astype(float))
    data['avg_vel'] = data[['AbsVelocity_X', 'AbsVelocity_Y']].mean(axis=1)
    # data['avg_vel'] = data.apply(lambda row: getAverageVelocity(data['AbsVelocity_X'], data['AbsVelocity_Y']), axis=1, result_type='expand')
    print("Got avg value!")
    print("Avg value: \n", data['avg_vel'].to_numpy())
    # data['avg_vel'] = data.apply(lambda row: \
    #     (abs(row.AbsVelocity_X.astype(float)) + \
    #     abs(row.AbsVelocity_Y.astype(float))).mean(), axis=1, result_type='expand')
    data = data.sort_values(by=['avg_vel'], ascending=True)

    x = np.arange(len(data))
    y = data['avg_vel'].to_numpy()
    print("Gen nrs: ", len(x))
    print("Data len: ", len(y))
    plt.bar(x, y)
    # plt.show()
    plt.savefig(visualFile)


def getAverageVelocity(velX, velY):
    # Abs Res: 6.233401909964011 1.3921506215067325
    # Rel res: 
    # data = pd.read_csv(dataFile, dtype='category')
    sub_X = abs(velX.astype(float))
    sub_Y = abs(velY.astype(float))
    sumXY = sub_X + sub_Y
    return sumXY / 2


def getNotWaitIds(waitIds, dataFile, outFile):
    wait = pd.read_csv(waitIds, dtype='category')
    data = pd.read_csv(dataFile, dtype='category')

    print("Read data files")
    wait_cols = wait[['ObjectId', 'csv_name']]
    wait_cols = wait_cols.values.tolist()

    all_cols = data[['ObjectId', 'csv_name']]
    all_cols = all_cols.values.tolist()

    not_wait_ids = []
    for i in all_cols:
        if i not in wait_cols:
            not_wait_ids.append(i)

    res = pd.DataFrame(not_wait_ids, columns=['ObjectId', 'csv_name'])
    res = res.drop_duplicates()
    res.to_csv(outFile)

    
    # all_ids = data[['ObjectId', 'Timestamp', 'csv_name']]
    # wait_ids = wait[['ObjectId', 'Timestamp', 'csv_name']]
    # not_wait_ids = all_ids.merge(wait_ids
    # , how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
    # print(not_wait_ids.head())
    # print(len(not_wait_ids))
    # not_wait_ids.to_csv(outFile)


def getDataWithoutWaitingRows(dataFile, ids, outFile):
    ids = pd.read_csv(ids, dtype='category')
    data = pd.read_csv(dataFile, dtype='category')
    data_cols = data[['ObjectId', 'Timestamp', 'csv_name']]
    data_cols = data_cols.values.tolist()
    print("Cols: ", data_cols[0])

    rows = []
    counter = 0
    for row in ids.values.tolist():
        row_val = row[1:4]
        if row_val in data_cols:
            print(counter, " ", row_val)
            rows.append(row)
            counter += 1
    res = pd.DataFrame(rows)
    res.to_csv(outFile)




def getDataWithoutWaiting(waitingFile, dataFile, outFile):
    wait = pd.read_csv(waitingFile, dtype='category')
    data = pd.read_csv(dataFile, dtype='category')

    print("Read data files")

    res = data.merge(wait, on=['ObjectId', 'csv_name', 'Timestamp'], left_index=True, right_index=True,
             how='left', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)').drop_duplicates()
    
    # print("All head:\n", df_all.head())
    # res = df_all[df_all['_merge'] == 'left_only']
    print("Res head:\n", res.head())
    print("Len by subtrackt: ", len(data) - len(wait))
    print("Len of result: ", len(res))
    # # print(res['Timestamp'])
    res.to_csv(outFile)


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


def saveWaitingSlowingSpeedingRecords(inFile, waitingFile, slowingFile, speedingFile):
    data = pd.read_csv(inFile, dtype='category')
    relX = data.loc[data['RelVelocity_X'].astype(float) == 0.0]
    relY = data.loc[data['RelVelocity_Y'].astype(float) == 0.0]
    absX = data.loc[data['AbsVelocity_X'].astype(float) == 0.0]
    absY = data.loc[data['AbsVelocity_Y'].astype(float) == 0.0]

    chained = relX.append([relY, absY, absX])
    chained = chained.drop_duplicates()

    data = data[data.columns.drop(list(data.filter(regex='Unnamed')))]
    chained = chained[chained.columns.drop(list(chained.filter(regex='Unnamed')))]
    print("Cols of waiting: ", chained.columns)

    # Remove all records where velocity is 0
    not_waiting_data = pd.concat([data, chained]).drop_duplicates(keep=False)

    # Set velocities to absolute
    not_waiting_data['AbsVelocity_X'] = not_waiting_data['AbsVelocity_X'].astype(float).abs()
    not_waiting_data['AbsVelocity_Y'] = not_waiting_data['AbsVelocity_Y'].astype(float).abs()
    not_waiting_data['RelVelocity_X'] = not_waiting_data['RelVelocity_X'].astype(float).abs()
    not_waiting_data['RelVelocity_Y'] = not_waiting_data['RelVelocity_Y'].astype(float).abs()

    # Add velocities to get speed
    not_waiting_data['abs_vel_sum'] = not_waiting_data['AbsVelocity_X'].astype(float) + \
                                        not_waiting_data['AbsVelocity_Y'].astype(float)
    not_waiting_data['rel_vel_sum'] = not_waiting_data['RelVelocity_X'].astype(float) + \
                                        not_waiting_data['RelVelocity_Y'].astype(float)

    abs_sum = not_waiting_data['abs_vel_sum'].sum()
    rel_sum = not_waiting_data['rel_vel_sum'].sum()
    lening =  len(not_waiting_data)
    abs_avg = abs_sum / lening
    rel_avg = rel_sum / lening

    print("Abs avg: ", abs_sum)
    print("Lenght: ", lening)
    print("Avg abs: ", abs_avg)
    print("Rel abs: ", rel_avg)

    # Filter out records, whose speed is lower than the average
    # abs2 = not_waiting_data.loc[not_waiting_data['abs_vel_sum'] <= 3.35]
    # rel2 = not_waiting_data.loc[not_waiting_data['rel_vel_sum'] <= 3.35]

    abs2 = not_waiting_data.loc[not_waiting_data['abs_vel_sum'] <= abs_avg]
    rel2 = not_waiting_data.loc[not_waiting_data['rel_vel_sum'] <= rel_avg]


    # relX2 = not_waiting_data.loc[data['RelVelocity_X'].astype(float) == 3.1]
    # relY2 = not_waiting_data.loc[data['RelVelocity_Y'].astype(float) == 0.7]
    # absX2 = not_waiting_data.loc[data['AbsVelocity_X'].astype(float) <= 3.1]
    # absY2 = not_waiting_data.loc[data['AbsVelocity_Y'].astype(float) <= 0.7]

    chained2 = rel2.append(abs2)
    chained2 = chained2.drop_duplicates()

    chained2 = chained2[chained2.columns.drop(list(chained2.filter(regex='Unnamed')))]

    not_waiting_data = pd.concat([not_waiting_data, chained2]).drop_duplicates(keep=False)
    print("Cols: ", not_waiting_data.columns)

    chained.to_csv(waitingFile)
    chained2.to_csv(slowingFile)
    not_waiting_data.to_csv(speedingFile)



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


def deleteExcessColumns(fileName, start, end, outFile):
    data = pd.read_csv(fileName, dtype='category')
    cols = len(data.columns)

    new_data = data.iloc[:, start:cols+end]
    new_data.to_csv(outFile)


def checkOverlapping(file1, file2):
    df1 = pd.read_csv(file1, dtype='category')
    df2 = pd.read_csv(file2, dtype='category')
    print("Len 1: ", len(df1))
    print("Len 2: ", len(df2))

    s1 = pd.merge(df1, df2, how='inner', on=['uniqueId'])
    ids = s1['uniqueId'].drop_duplicates()
    # print(s1)
    print(len(ids))


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
    # saveWaitingPandasIds("datasets/intersections-dataset-before-thresh.csv", 'datasets/waitingIds.csv')
    # saveWaitingTracks('datasets/not-waiting-ids.csv', "datasets/intersections-dataset-before-thresh.csv", 'datasets/not-waiting-thresh.csv')
    # getDataWithoutWaiting('datasets/waiting-thresh.csv', "datasets/intersections-dataset-before-thresh.csv", "datasets/not-waiting-thresh5.csv")
    # getDataWithoutWaitingRows("datasets/intersections-dataset-before-thresh.csv", "datasets/not-waiting-ids.csv", "datasets/not-waiting-thresh5.csv")
    # splitIds("datasets/dataset-without-waiting.csv", "datasets/not-waiting-split-ids.csv", 328)
    # saveWaitingTracks("datasets/waitingIds.csv", 'datasets/intersections-dataset-before-thresh.csv', 'datasets/waiting-thresh-02.csv')
    # getBeforeThreshold("datasets/dataset-without-waiting.csv", "datasets/not-waiting-thresh.csv")
    # deleteExcessColumns("datasets/speeding-thresh-peers2.csv", 2, 0, "datasets/speeding-thresh-peers3.csv")
    # getNotWaitIds('datasets/waiting-thresh.csv', "datasets/intersections-dataset-before-thresh.csv", "datasets/not-waiting-ids2.csv")
    # sumMean = getAverageVelocity("datasets/intersections-dataset-before-thresh.csv")
    # saveWaitingSlowingSpeedingRecords("datasets/feb/intersections-dataset-before.csv", 
    #                                 "datasets/feb/waiting-before-thresh.csv",
    #                                 "datasets/feb/slowing-before-thresh.csv",
    #                                 "datasets/feb/speeding-before-thresh.csv")
    # testLeftJoin()
    # print("Abs: ", sumMean)
    # checkOverlapping("datasets/speeding-before-thresh.csv", "datasets/waiting-before-thresh.csv")
    # visualizeAvgVelocity("datasets/feb/intersections-dataset-before.csv", "datasets/feb/avg_vel.png")
if __name__ == "__main__":
    main()




# Currently I have full trajectories of those vehicles, that waited at some point
# But, to make the snapshow data, I need to have those records that have velocity 0, and those that have other vehicles at the same time at the intersection
# Snapshot:
# - Velocity is 0
# There are other vehicles at the roundabout
# The other vehicle's velocity is not 0