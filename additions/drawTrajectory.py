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
    # print("Hey")
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
    # plotThreshLines(data.csv_name.iloc[0]) # Assume there's only one csv_name

    # ids = data.uniqueId.unique()
    # ids = ['69574', '68978', '69460']
    # ids = ['11936', '11757', '11853']
    # ids = ['14805', '14915', '15110']
    # ids = ['140553']
    # ids = ['15110']
    # colors = cm.rainbow(np.linspace(0, 1, len(ids)))


    # for i in range(len(ids)):
    for i in range(len(ids)):
        sub = data.loc[data['uniqueId'] == ids[i]]
        print("Csv: ", sub.csv_name.unique())

        # if len(timestamps) == 0:
        #     timestamps = sub.Timestamp.unique()
        # else:
        #     steps = set(timestamps)
        #     res = steps.intersection(sub.Timestamp)
        #     timestamps = list(res)
        # Mark the first location
        xS = round(float(sub.iloc[1]['relative_x_trans']), 8)
        yS = round(float(sub.iloc[1]['relative_y_trans']), 8)
        # plt.plot(xS, yS, color='blue', marker="o")
        # Round all the coordinates
        # x = round(sub['relative_x_trans'].astype(float), 3)
        # y = round(sub['relative_y_trans'].astype(float), 3)

        # vel_x = round(sub['RelVelocity_X'].astype(float), 3)
        # vel_y = round(sub['RelVelocity_Y'].astype(float), 3)

        # waits_x, waits_y = [], []
        # speeds_x, speeds_y = [], []
        # slows_x, slows_y = [], []

        # for j, k, m, n in zip(vel_x, vel_y, x, y):
        #     # print(j, k)
        #     summa = abs(j) + abs(k)

        #     if j == 0 or k == 0:
        #         # print("Waiting!")
        #         waits_x.append(m)
        #         waits_y.append(n)
        #     elif summa < 7.22:
        #         slows_x.append(m)
        #         slows_y.append(n)
        #     else:
        #         speeds_x.append(m)
        #         speeds_y.append(n)

        # print("Match? ", len(vel_x), len(waits_x) + len(slows_x) + len(speeds_x))



        # # Mark the first location
        xS = round(float(sub.iloc[1]['relative_x']), 8)
        yS = round(float(sub.iloc[1]['relative_y']), 8)
        # plt.plot(xS, yS, color='blue', marker="o")
        # # Round all the coordinates
        x = round(sub['relative_x'].astype(float), 3)
        y = round(sub['relative_y'].astype(float), 3)

        plt.scatter(xS, yS, s=20, color=colors[i])
        plt.scatter(x,y, color=colors[i], s=5)

        
        # plt.scatter(speeds_x, speeds_y, color='green', s=2)
        # plt.scatter(slows_x, slows_y, color='orange', s=2)
        # plt.scatter(waits_x, waits_y, color='red', s=4)
        # plt.scatter(x,y, color=colors[i], s=3)
        plt.axis('equal')
        # plt.plot(x,y, color=colors[i])
    plt.plot(centerX, centerY, color='red', marker="o")
    
    # print("Timestamps are: ", timestamps)
    # steps = data.loc[np.isin(data['Timestamp'].to_numpy(), timestamps)]

    # relX = list(round(steps.relative_x_trans.astype(float), 3))
    # relY = list(round(steps.relative_y_trans.astype(float), 3))
    # print("RelX:\n", relX)
    # print("RelY:\n", relY)

    sub_1 = data.loc[data.uniqueId == ids[0]]
    sub_2 = data.loc[data.uniqueId == ids[1]]
    sub_3 = data.loc[data.uniqueId == ids[2]]

    times_1 = sub_1.Timestamp.unique()
    times_2 = sub_2.Timestamp.unique()
    times_3 = sub_3.Timestamp.unique()
    print("Times 1:\n", times_1[:5])

    times = set(times_1).intersection(times_2)
    times = list(times)
    print("First intersect: ", times)
    times_all = set(times).intersection(times_3)
    times_all = list(times_all)

    sub_inter = data[np.isin(data['Timestamp'].to_numpy(), times_all)]
    relX = round(sub_inter.relative_x_trans.astype(float), 3)
    relY = round(sub_inter.relative_y_trans.astype(float), 3)

    print("RelX:\n", relX)
    print("RelY:\n", relY)
    plt.scatter(relX, relY, color='black', s=2)

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

