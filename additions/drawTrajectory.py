import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def threshPoints(csv_name):
    # Return the two points from what to make the line for every direction
    # Then use the line to determine if other points are above or below it
    # North, east, south, west, north
    # Detemine if point should be left or right from line by comparing with center point!
    if 'roslyn' in csv_name:
        return np.array([
           [-3, 20],
           [7, 10],
           [-3, 0],
           [-13, 10],
           [-3, 20]
        ])
    elif 'leith' in csv_name:
        return np.array([
            [-8, 20],
            [9, 18],
            [7, 0],
            [-10, 2],
            [-8, 20]
        ])
    elif 'oliver' in csv_name:
        return np.array([
            [-7, 21],
            [9, 19],
            [7, 1],
            [-9, 3],
            [-7, 21]
        ])
    elif 'orchard' in csv_name:
        return np.array([
            [-9, 19],
            [12, 17],
            [10, -2],
            [-11, 0],
            [-9, 19]
        ])
    elif 'queen' in csv_name:
        return np.array([
            [-13, 21],
            [9, 20],
            [8, 0],
            [-14, 1],
            [-13, 21]
        ])


def plotThreshLines(csv_name):
    points = threshPoints(csv_name)
    points_x = points[:,0]
    points_y = points[:,1]
    plt.plot(points_x, points_y, color='black')


def plotTrajectory(data, fileName, centerX, centerY):
    print("Hey")
    # data = data.loc[data.csv_name.str.contains("oliver")]
    # dirs = data.origin.unique()
    # print(dirs)
    # newData = pd.DataFrame(columns=data.columns)
    # idd = []
    # for i in dirs:
    #     sub = data.loc[data.origin == i]
    #     # print("Sub: \n", sub.head())
    #     # newData = newData.append(sub.head(5), ignore_index=True)
    #     ids = sub.uniqueId.unique()
    #     idd.append(ids[:5])
    
    # idd = [item for sublist in idd for item in sublist]
    # print("Idd: \n", idd)
    # newData = data[data['uniqueId'].isin(idd)]
    # # for i in idd:
    # #     print("i: ", i)
    # #     newData = newData.append(data.loc[data.uniqueId == i])
    # # newData = data.loc[data.uniqueId in idd]
    # print("New data: ", newData)
    # data = newData
    # print("Filtered data!")
    # print("Columns: \n", data.columns)
    # print("Csv name: \n", data.csv_name.head(5))
    # plotThreshLines(data.name.iloc[0])
    plotThreshLines(data.csv_name.iloc[0]) # Assume there's only one csv_name

    ids = data.uniqueId.unique()
    colors = cm.rainbow(np.linspace(0, 1, len(ids)))

    # for i in range(len(ids)):
    for i in range(len(ids)):
        sub = data.loc[data['uniqueId'] == ids[i]]
        # Mark the first location
        # xS = round(float(sub.iloc[1]['relative_x_trans']), 8)
        # yS = round(float(sub.iloc[1]['relative_y_trans']), 8)
        # plt.plot(xS, yS, color='blue', marker="o")
        # Round all the coordinates
        x = round(sub['relative_x_trans'].astype(float), 3)
        y = round(sub['relative_y_trans'].astype(float), 3)


        # # Mark the first location
        # xS = round(float(sub.iloc[1]['relative_x']), 8)
        # yS = round(float(sub.iloc[1]['relative_y']), 8)
        # plt.plot(xS, yS, color='blue', marker="o")
        # # Round all the coordinates
        # x = round(sub['relative_x'].astype(float), 3)
        # y = round(sub['relative_y'].astype(float), 3)

        # plt.scatter(xS, yS, s=20, color=colors[i])
        # plt.scatter(x,y, color=colors[i], s=5)
        plt.scatter(x,y, color=colors[i], s=3)
        plt.axis('equal')
        # plt.plot(x,y, color=colors[i])
    plt.plot(centerX, centerY, color='red', marker="o")

    # north, south, east, west = thresholds
    # plt.plot([west, east], [south, south], color='black')
    # plt.plot([west, east], [north, north], color='black')
    # plt.plot([west, west], [north, south], color='black')
    # plt.plot([east, east], [north, south], color='black')

    # plt.show()
    plt.axis('equal')
    plt.savefig(fileName, transparent=True)
    plt.close()

# # data = pd.read_csv("../../records/records_0-5000.csv", dtype='category')
# # data = pd.read_csv("datasets/intersections-dataset-transformed.csv", dtype='category')
# data = pd.read_csv("datasets/testing/trans-north.csv", dtype='category')
# # north = data.loc[data.origin == 'north']
# # south = data.loc[data.origin == 'south']
# # east = data.loc[data.origin == 'SE']
# # west = data.loc[data.origin == 'NW']

# centerX, centerY = -2.5, 8
# thresholds = [20, -3, 3, -7]
# plotTrajectory(data, "", centerX, centerY, thresholds)

