import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plotTrajectory(data, fileName, centerX, centerY, thresholds):
    # data = pd.read_csv("intersections-dataset.csv", dtype='category')

    # x = []
    # y = []
    ids = data['ObjectId'].unique()

    colors = cm.rainbow(np.linspace(0, 1, len(ids)))

    # for i in range(len(ids)):
    for i in range(len(ids)):
        sub = data.loc[data['ObjectId'] == ids[i]]
        # sub = sub.sort_values(by=['Timestamp'])
        print(len(sub))
        # Mark the first location
        # xS = round(float(sub.iloc[1]['relative_x']), 3)
        # yS = round(float(sub.iloc[1]['relative_y']), 3)
        # Round all the coordinates
        x = round(sub['relative_x'].astype(float), 3)
        y = round(sub['relative_y'].astype(float), 3)

        # plt.scatter(xS, yS, s=20, color=colors[i])
        # plt.scatter(x,y, color=colors[i], s=5)
        plt.scatter(x,y, color=colors[i], s=2)
        # plt.plot(x,y, color=colors[i])
    plt.plot(centerX, centerY, color='red', marker="o")

    north, south, east, west = thresholds
    plt.plot([west, east], [south, south], color='black')
    plt.plot([west, east], [north, north], color='black')
    plt.plot([west, west], [north, south], color='black')
    plt.plot([east, east], [north, south], color='black')

    # plt.show()
    plt.savefig(fileName, transparent=True)
    plt.close()

# data = pd.read_csv("../../records/records_0-5000.csv", dtype='category')
# plotTrajectory(data, '.tracks')

