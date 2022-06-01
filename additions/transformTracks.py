import pandas as pd
import matplotlib.pyplot as plt
from drawTrajectory import plotTrajectory, threshPoints
from os import listdir
from os.path import isfile, join
import math
import numpy as np


def transform(origin, point, angle):
    """
    Rotate a point clockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def getCenterPoint():
    return -2.5, 10


def rotate(x, y, rad):
    res_x = math.cos(rad) * x - math.sin(rad) * y
    res_y = math.sin(rad) * x + math.cos(rad) * y
    return res_x, res_y


def transformRow(relX, relY, origin, destination, csv_name):
    relX = float(relX)
    relY = float(relY)

    center = getCenterPoint()
    orig_point = relX, relY
    # Transform all the tracks so that north is 180 from origin etc
    if origin == 'north':
        relX, relY = transform(center, orig_point, math.radians(180))
    elif origin == 'south':
        relX, relY = transform(center, orig_point, math.radians(0))
    elif origin == 'east':
        relX, relY = transform(center, orig_point, math.radians(90))
    elif origin == 'west':
        relX, relY = transform(center, orig_point, math.radians(-90))
    elif origin == 'SE':
        relX, relY = transform(center, orig_point, math.radians(45))
    elif origin == 'NW':
        relX, relY = transform(center, orig_point, math.radians(-135))
    elif origin == 'NE':
        relX, relY = transform(center, orig_point, math.radians(135))

    if 'leith' in csv_name:
        return rotate(relX, relY, math.radians(-8))
    elif 'queen' in csv_name:
        return rotate(relX, relY, math.radians(-4))
    elif 'oliver' in csv_name:
        return rotate(relX, relY, math.radians(-13))
    elif 'orchard' in csv_name:
        return rotate(relX, relY, math.radians(-13))
    elif 'roslyn' in csv_name:
        return rotate(relX, relY, math.radians(5))

def replaceName(csv_name):
    if "leith" in csv_name:
        return "leith"
    elif 'queen' in csv_name:
        return 'queen'
    elif 'orchard' in csv_name:
        return 'orchard'
    elif 'oliver' in csv_name:
        return 'oliver'
    elif 'roslyn' in csv_name:
        return 'roslyn'

def getDist(x1, x2, y1, y2):
    return math.hypot(x1 - x2, y1 - y2)


def getThreshDist(dist, origin, name, closeDict):
    thresh = closeDict[name + "-" + origin]
    # thresh = closestFirstPoints.loc[(closestFirstPoints.origin == origin) & \
    #                                  (closestFirstPoints.name == name)]
    # print("Thresh: ", thresh)
    return thresh


def insideOrOutsideThresh(dist, distThresh):
    return dist <= distThresh

def getClosestDict(closestDataset):
    names = list(closestDataset.name)
    origs = list(closestDataset.origin)
    tuples = zip(names, origs)

    new_list = [i[0] + "-" + i[1] for i in tuples]
    d = dict(zip(new_list, list(closestDataset.dist)))
    # print("D: \n", d)
    return d


def alignTracksClosestStart(fileIn, fileOut):
    data = pd.read_csv(fileIn, dtype='category')
    print("Read data!")
    roundabouts = ['leith', 'queen', 'oliver', 'orchard', 'roslyn']
    # Get the furthest point for each intersection and direction
    # Calculate their distance
    # Select the row with the shortest distance
    # Only select part of track that comes after it
    x, y = getCenterPoint()
    data['name'] = data.apply(lambda row : replaceName(row['csv_name']), axis=1, result_type='expand')
    data['dist'] = data.apply(lambda row: getDist(x, float(row['relative_x_trans']), y, float(row['relative_y_trans'])), axis=1, result_type='expand')
    print("Data columns: ", data.columns)
    furIdx = data.groupby(['uniqueId'])['dist'].transform(max) == data['dist']
    # furthestPoints = data[furIdx]
    # furIdx = data.groupby('uniqueId')['dist'].max()
    # print("Ids: ", furIdx2)
    # print("Ids,not reindexed: ", data.groupby('uniqueId')['dist'].max().head(5))

    furthestPoints = data[furIdx]

    # furthestPoints.to_csv("datasets/feb/furthestPoints1.csv")
    furthestPoints.to_csv("datasets/feb/furthestPoints.csv")
    print("Furthest points: \n", furthestPoints.head())
    
    closeIdx = furthestPoints.groupby(['origin', 'name'])['dist'].transform(min) == furthestPoints['dist']
    # closeIdx = data.groupby(['origin', 'name'])['dist'].min()
    closestFirstPoints = furthestPoints[closeIdx]
    print("Closest points: \n", closestFirstPoints)
    closestFirstPoints.to_csv("datasets/feb/closestPoints.csv")

    # closeDict = dict(zip(closestFirstPoints.name, closestFirstPoints.origin, closestFirstPoints.dist))
    
    # closeDict = closestFirstPoints[['name', 'origin', 'dist']].to_dict(orient='records')
    closeDict = getClosestDict(closestFirstPoints)
    print("Close dist: \n", closeDict)

    data['threshDist'] = data.apply(lambda row: getThreshDist(float(row['dist']), row['origin'], row['name'], closeDict), axis=1, result_type='expand')
    print("Dist head: \n", data.head(5))
    data.to_csv("datasets/feb/aligned-thresh.csv")
    data['insideThresh'] = data.apply(lambda row: insideOrOutsideThresh(float(row['dist']), float(row['threshDist'])), axis=1, result_type='expand')
    print("Inside thresh: \n", data.head(5))

    res = data.loc[data.insideThresh == True]
    print("Res length: ", len(res))
    print("Res head: \n", res.head(5))
    res.to_csv(fileOut)


    
    # newData = pd.DataFrame()
    # for name in roundabouts:
    #     sub = data.loc[data.name == name]
    #     directions = sub.origin.unique()
    #     print("Directions: \n", directions)
    #     for j in directions:
    #         sub2 = sub.loc[sub.origin == j]
    #         print("Sub: \n", sub2.head(5))
    #         thresh = closestFirstPoints.loc[(closestFirstPoints.origin == j) & (closestFirstPoints.name == name)]
    #         print("Thresh is: \n", thresh)
    #         thresh_dist = float(thresh.dist)
    #         print("Thresh dist: ", thresh_dist)
    #         res = sub2.loc[sub2.dist <= thresh_dist]
    #         print("** Res head: \n", res.head())
    #         res.to_csv("datasets/feb/aligned-" + name + "-" + j + ".csv")
            # df_mask = sub2['dist'] <= thresh_dist
            # print("** Masked df size: ", len(df_mask))
            # newData = newData.append(sub2[df_mask])
            # df_mask.to_csv("datasets/feb/aligned-" + name + "-" + j + ".csv")
    
    # newData.to_csv(fileOut)


def beforeThresh(x, y, csv_name, origin):
    a, b, c, d, _ = threshPoints(csv_name)
    p = np.array([x, y])

    if origin == 'north': # Left of thresh, a-b
        return np.cross(p-a, b-a) < 0
    elif origin == 'east': # Right of thresh, b-c
        return not np.cross(p-b, c-b) < 0
    elif origin == 'south': # Right of thresh, c-d
        return np.cross(p-c, d-c) < 0
    elif origin == 'west': # Left of thresh, a-d
        return np.cross(p-a, d-a) < 0
    elif origin == 'NW': # Left of thresh, a-d
        return not np.cross(p-a, d-a) < 0
    elif origin == 'SE': # Right of thresh, b-c
        return np.cross(p-b, c-b) < 0
    elif origin == 'NE': # Right of thresh, a-b
        return not np.cross(p-a, b-a) < 0


def getBeforeThresh(fileIn, outFile):
    data = pd.read_csv(fileIn, dtype='category')
    data['before'] = data.apply(lambda row: \
                    beforeThresh(float(row['relative_x_trans']), \
                                float(row['relative_y_trans']), \
                                row['csv_name'], \
                                row['origin']), axis=1, result_type='expand')
    beforeData = data.loc[data.before == True]
    beforeData.drop(columns=['before'])
    beforeData.to_csv(outFile)


def transformDataLines(fileName, outFile):
    data = pd.read_csv(fileName, dtype='category')

    data[['relative_x_trans', 'relative_y_trans']] = \
        data.apply(lambda row : transformRow(row['relative_x'], \
            row['relative_y'], row['origin'], row['destination'], row['csv_name']), \
                axis=1, result_type='expand')
    # data[['roundabout_thresh', 'furthest_thresh']] = createThresh(data)
    data.to_csv(outFile)


def plotData(data, saveName, csvName):
    # data = pd.read_csv(fileName, dtype='category')
    # data = data.loc[(data.csv_name.str.contains("leith")) & (data.origin == 'north')]
    # data = data.loc[(data.csv_name.str.contains(csvName))]
    # ids = data.uniqueId.unique()
    # ids = ids[:10]
    # ids = ['3', '4', '39', '51', '72', '82', '97', '112', '134', '138', '153', '159', '162', '198', '258', '280', '287', '295', '296', '323']
    # print("Ids total: ", len(data.uniqueId.unique()))
    # ids = list(data.uniqueId.unique())[:20]
    # print("Ids: ", ids)
    # data = data.loc[data.apply(lambda x: x.uniqueId in ids, axis=1, result_type='expand')]
    # print("Read data!")
    centerX, centerY = getCenterPoint()
    plotTrajectory(data, saveName, centerX, centerY)



def main():
    # plotTransformed("oliver05.csv", 
    #             "oliver", 
    #             '../plots/Oliver05Before22.png')
    # plotTransformed("../../records/records_485000-490000.csv", 
    #                 "leith-croydon", 
    #                 '../plots/LeithTracksThresh.png')
    # plotTransformed("../../records/records_2160000-2165000.csv", 
    #             "urban-stationary-oliver-wyndora", 
    #             '../plots/OliverTracksThresh.png')
    # plotTransformed("../../records/records_1560000-1565000.csv", 
    #         "urban-stationary-queen-hanks", 
    #         '../plots/QueenTracksThresh.png')
    # plotTransformed("../../records/records_1870000-1875000.csv", 
    #         "urban-stationary-roslyn-crieff", 
    #         '../plots/RoslynTracksThresh.png')
    # plotTransformed("../../records/records_2955000-2960000.csv", 
    #         "urban-stationary-orchard-mitchell", 
    #         '../plots/OrchardTracksThresh.png')


    # Roundabout names                      latitude, longitude
    # leith-croydon                         -33.899, 151.111        records_485000-490000.csv
    # urban-stationary-oliver-wyndora       -33.7746, 151.284       records_2160000-2165000.csv
    # urban-stationary-queen-hanks          -33.9036, 151.127       records_1595000-1600000.csv
    # urban-stationary-roslyn-crieff        -33.9005, 151.114       records_1870000-1875000.csv
    # urban-stationary-orchard-mitchell     -33.7654, 151.274       records_2955000-2960000.csv
    # print()

    # transformData("oliver05.csv", "oliver", "oliver05Transform.csv")
    # plotData("oliver05Before.csv", "../plots/oliver05Before.png")
    # transformData("../../records/records_0-5000.csv", "../../records/records_0-5000Trans.csv")
    # getFilesTrans("../../records/")
    # readFilesIntoOne("../../records/transformed/", "../../records/transformed/intersections-dataset-transformed.csv")
    # transformDataLines("../../intersections-dataset.csv", "../../intersections-dataset-transformed.csv")
    # calculateThresholds("datasets/intersections-dataset-transformed.csv")


    # transformDataLines("../../records/records_25000-30000.csv", "datasets/testing/leith.csv")
    # plotData("datasets/testing/leith.csv", 'datasets/testing/leith-tracks.png')

    # transformDataLines("../../records/records_1310000-1315000.csv", "datasets/testing/queen.csv")
    # plotData("datasets/testing/queen.csv", 'datasets/testing/queen-tracks.png')

    # transformDataLines("../../records/records_2585000-2590000.csv", "datasets/testing/oliver.csv")
    # plotData("datasets/testing/oliver.csv", 'datasets/testing/oliver-tracks.png')

    # transformDataLines("../../records/records_3020000-3025000.csv", "datasets/testing/orchard.csv")
    # plotData("datasets/testing/orchard.csv", 'datasets/testing/orchard-tracks.png')
    
    # transformDataLines("../../records/records_1930000-1935000.csv", "datasets/testing/roslyn.csv")
    # plotData("datasets/testing/roslyn.csv", 'datasets/testing/roslyn-tracks.png')
    # plotData("../../records/records_1840000-1845000.csv", "")

    # transformDataLines("datasets/intersections-dataset.csv", "datasets/feb/intersections-dataset-transformed.csv")
    # getBeforeThresh("datasets/feb/intersections-dataset-transformed.csv", "datasets/feb/intersections-dataset-before.csv")
    # plotData("datasets/feb/intersections-dataset-before.csv", "datasets/feb/before-oliver.png")
    # getFurthestThrest("datasets/testing/leith.csv")
    # alignTracksClosestStart("datasets/feb/intersections-dataset-transformed.csv", "datasets/feb/intersections-dataset-transformed-aligned.csv")
    # plotData("datasets/feb/intersections-dataset-transformed.csv", "datasets/feb/leith-north-not-aligned.png")
    # beforeFile = "datasets/feb/intersections-dataset-before.csv"
    # transFile = "datasets/feb/intersections-dataset-transformed.csv"

    # transData = pd.read_csv(transFile, dtype='category')
    # # beforeData = pd.read_csv(beforeFile, dtype='category')
    # # plotData(beforeData, "datasets/feb/plots/queen-before.png", 'queen')
    # plotData(transData, "datasets/april/plots/surrounds-intersect2.png", '')

if __name__ == "__main__":
    main()
