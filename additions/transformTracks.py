import pandas as pd
import matplotlib.pyplot as plt
from drawTrajectory import plotTrajectory
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

    if 'leith' in csv_name:
        return rotate(relX, relY, math.radians(-8))
    elif 'queen' in csv_name:
        return rotate(relX, relY, math.radians(-4))
    elif 'oliver' in csv_name:
        return rotate(relX, relY, math.radians(-13))
    elif 'orchard' in csv_name:
        return rotate(relX, relY, math.radians(-13))

# ROTATE THE THRESHOLDS!!!!!
def getCenterPoint():
    return -2.5, 10


def getThreshold():
    north = 17
    south = 5
    west = -7
    east = 6
    return north, south, east, west

# Function for calculating the rounabout threshold
def calculateThresholds(fileName):
    data = pd.read_csv(fileName, dtype='category')

    north = data.loc[data.origin == "north"]
    south = data.loc[data.origin == "south"]
    east = data.loc[data.origin == "SE"]
    west = data.loc[data.origin == "NW"]

    # Get 3 first points for each direction
    n3 = north.iloc[:3]
    s3 = south.iloc[:3]
    e3 = east.iloc[:3]
    w3 = west.iloc[:3]

    # Get average for each direction
    north_avg_x = n3.relative_x_trans.astype(float).mean()
    south_avg_x = s3.relative_x_trans.astype(float).mean()
    east_avg_y = e3.relative_y_trans.astype(float).mean()
    west_avg_y = w3.relative_y_trans.astype(float).mean()

    x_east = east.relative_x_trans.astype(float)
    y_east = east.relative_y_trans.astype(float)
    x_west = west.relative_x_trans.astype(float)
    y_west = west.relative_y_trans.astype(float)

    plt.scatter(x_east, y_east, color='red', s=2)
    # plt.scatter(x_west, y_west, color='blue')
    plt.show()

    # print("North: ")
    # getThresh(north.iloc[3:], 'x', north_avg_x)
    # print("South: ")
    # getThresh(south.iloc[3:], 'x', south_avg_x)
    # print("East: ")
    # d = east.loc[east.uniqueId == '76730']
    # a = d.relative_y_trans.astype(float)[:3].mean()
    # getThresh(d.iloc[3:], 'y', a)
    # print("West: ")
    # getThresh(west.iloc[3:], 'y', east_avg_y)

    # Accorfing to this:
    # North: 18
    # South: -6
    # 


def getThresh(data, axis, avg):
    count = 0
    if axis == 'x':
        for i, j in zip(data.relative_x_trans.astype(float), data.relative_y_trans.astype(float)):
            diff = abs(avg - i)
            if count < 200:
                print("X diff for north: ", diff, " ", j) # From this result I assume that when the diff is larger than 1, it has passed the threshold
            count += 1
            #  if diff > 1:
            #     print("y value is: ", j)
            #     return j
    else:
        for i, j in zip(data.relative_y_trans.astype(float), data.relative_x_trans.astype(float)):
            diff = abs(avg - i)
            if count < 200: 
                print("Diff: ", diff, " ", j) # From this result I assume that when the diff is larger than 1, it has passed the threshold
            count += 1
            # if diff > 1:
            #     print("x value is: ", j)
            #     return j


def transformData(fileName, outFile):
    data = pd.read_csv(fileName, dtype='category')
    files = data['csv_name'].unique()

    res = pd.DataFrame()
    for i in files:
        sub = data.loc[data['csv_name'] == i]
        ids = sub['ObjectId'].unique()
        for j in ids:
            track = sub.loc[sub['ObjectId'] == j]
            k = transform(track, track.iloc[0]['origin'], track.iloc[0]['destination'])
            res = res.append(k)
    res.columns = data.columns
    res.to_csv(outFile)


def transformDataLines(fileName, outFile):
    data = pd.read_csv(fileName, dtype='category')

    data[['relative_x_trans', 'relative_y_trans']] = \
        data.apply(lambda row : transformRow(row['relative_x'], \
            row['relative_y'], row['origin'], row['destination'], row['csv_name']), \
                axis=1, result_type='expand')
    data.to_csv(outFile)


def plotData(fileName, saveName):
    data = pd.read_csv(fileName, dtype='category')
    centerX, centerY = getCenterPoint()
    thresholds = getThreshold()
    plotTrajectory(data, saveName, centerX, centerY, thresholds)


def getFilesTrans(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files = [path + f for f in files]
    for i in files:
        range_ = i.split("_")[1]
        range_ = range_.split(".")[0]
        outFile = path + "transformed/" + "records_" + range_ + "Trans.csv"
        print("File: ", outFile)
        transformData(i, outFile)


def readFilesIntoOne(path, outFile):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files = [path + f for f in files]
    out = open(outFile, 'a')
    
    firstFile = open(files[0], 'r')
    firstLine = firstFile.readline()
    out.write(firstLine)

    for i in files:
        print("File: ", i)
        with open(i, 'r') as f:
            rows = f.readlines()[1:]
            res = ''.join(rows)
            out.write(res)
            f.close()
    out.close()


def main():
    # plotTransformed("oliver05.csv", 
    #             "oliver", 
    #             '../plots/Oliver05Before2.png')
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

    transformDataLines("../../records/records_3020000-3025000.csv", "datasets/testing/orchard.csv")
    plotData("datasets/testing/orchard.csv", 'datasets/testing/orchard-tracks.png')
    # plotData("../../records/records_1840000-1845000.csv", "")

if __name__ == "__main__":
    main()
