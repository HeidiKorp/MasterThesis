import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plotTrajectory(data, fileName, centerX, centerY, thresholds):
    # x = []
    # y = []
    # ids = data['ObjectId'].unique()
    ids = data.uniqueId.unique()
    # print(ids)
    # print("Len: ", len(ids))
    # for i in range(len(ids)):
    #     print("i: ", i, " id: ", ids[i])
    # print("239 is: ", ids[239])
    # print(data.loc[data.uniqueId == ids[239]])

    colors = cm.rainbow(np.linspace(0, 1, len(ids)))

    # for i in range(len(ids)):
    for i in range(len(ids)):
        sub = data.loc[data['uniqueId'] == ids[i]]
        # Mark the first location
        xS = round(float(sub.iloc[1]['relative_x_trans']), 8)
        yS = round(float(sub.iloc[1]['relative_y_trans']), 8)
        plt.plot(xS, yS, color='blue', marker="o")
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
        plt.scatter(x,y, color=colors[i], s=1)
        plt.axis('equal')
        # plt.plot(x,y, color=colors[i])
    plt.plot(centerX, centerY, color='red', marker="o")

    north, south, east, west = thresholds
    plt.plot([west, east], [south, south], color='black')
    plt.plot([west, east], [north, north], color='black')
    plt.plot([west, west], [north, south], color='black')
    plt.plot([east, east], [north, south], color='black')

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

