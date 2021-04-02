import pandas as pd
import matplotlib.pyplot as plt
from drawTrajectory import plotTrajectory
from os import listdir
from os.path import isfile, join


def transform(data, origin, dest):
    # Origin: north, destination: east
    # Y coordinate is the opposite of the center point (abs dist -> center point plus abs)
    # X coordinate is opposite of abs dist center point plus abs
    newData = data.copy()
    newData['relative_x'] = newData['relative_x'].astype(float)
    newData['relative_y'] = newData['relative_y'].astype(float)

    centX, centY = getCenterPoint()
    relY = []
    relX = []
    for index, row in newData.iterrows():
        newX = 0
        newY = 0
        diffY = abs(abs(centY) - abs(row['relative_y']))
        diffX = abs(abs(centX) - abs(row['relative_x']))
        if centY < 0 and row['relative_y'] > 0 or centY > 0 and row['relative_y'] < 0:
            diffY = abs(abs(centY) + abs(row['relative_y']))
        if centX < 0 and row['relative_x'] > 0 or centX > 0 and row['relative_x'] < 0:
            diffX = abs(abs(centX) + abs(row['relative_x']))
    

        if origin != "south":
            newX = diffX + centX
            newY = diffY + centY
            if row['relative_x'] > centX:
                newX = -diffX + centX
            if row['relative_y'] > centY:
                newY = -diffY + centY
        
        relY.append(newY)
        relX.append(newX)


    if origin != "south":
        newData['relative_y'] = relY
        newData['relative_x'] = relX

    return newData


def getCenterPoint():
    return -2.5, 8


def getThreshold():
    north = 20
    south = -3
    west = -7
    east = 3
    return north, south, east, west


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


def plotData(fileName, saveName):
    data = pd.read_csv(fileName, dtype='category')
    centerX, centerY = getCenterPoint()
    thresholds = getThreshold()
    plotTrajectory(data, saveName, centerX, centerY, thresholds)


def getFilesTrans(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for i in files:
        range_ = i.split("_")[1]
        range_ = range_.split(".")[0]
        print("File: ", "records_" + range_ + "Trans.csv")
        transformData(i, "records_" + range_ + "Trans.csv")


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

    # transformData("oliver05.csv", "oliver", "oliver05Transform.csv")
    # plotData("oliver05Before.csv", "../plots/oliver05Before.png")
    # transformData("../../records/records_0-5000.csv", "../../records/records_0-5000Trans.csv")
    getFilesTrans("../../records/")
if __name__ == "__main__":
    main()