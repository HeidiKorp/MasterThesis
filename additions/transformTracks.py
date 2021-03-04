import pandas as pd
import matplotlib.pyplot as plt
from drawTrajectory import plotTrajectory

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
        
        # if origin == "north" and dest == "east":
        #     newX = diffX + centX
        #     newY = diffY + centY
        #     if row['relative_x'] > centX:
        #         newX = -diffX + centX
        #     if row['relative_y'] > centY:
        #         newY = -diffY + centY
        # elif origin == "north" and dest == "south":
        #     newX = diffX + centX
        #     newY = diffY + centY
        #     if row['relative_x'] > centX:
        #         newX = -diffX + centX
        #     if row['relative_y'] > centY:
        #         newY = -diffY + centY


    if origin != "south":
        newData['relative_y'] = relY
        newData['relative_x'] = relX
    return newData


def getCenterPoint():
    return -3, 8



def main():
    data = pd.read_csv("../../records/records_2030000-2035000.csv", dtype='category')
    # data = pd.read_csv("../../intersections-dataset.csv", dtype='category')
    ids = data['ObjectId'].unique()
    sub = data[data['csv_name'].str.contains("urban-stationary-oliver-wyndora")]



    # print(ids[10:20])
    res = pd.DataFrame()
    for i in range(len(ids)):
        track = data.loc[data['ObjectId'] == ids[i]]
        k = transform(track, track.iloc[0]['origin'], track.iloc[0]['destination'])
        res = res.append(k)
    # print("Type: ", type(res))
    # print(res.head())
    plotTrajectory(res, 'transformedOliver.jpg')
    # sub = data.loc[data['ObjectId'] == "92"]
    # trans = transform(sub, sub.iloc[0]['origin'], sub.iloc[0]['destination'])
    # print("Origin: ", sub.iloc[0]['origin'], " destination", sub.iloc[0]['destination'], " id: ", sub.iloc[0]['ObjectId'], sub.iloc[0]['csv_name'])
    
    # # Mark the first coordinate
    # xS = round(float(sub.iloc[1]['relative_x']), 3)
    # yS = round(float(sub.iloc[1]['relative_y']), 3)
    # # Round all the coordinates
    # x = round(sub['relative_x'].astype(float), 3)
    # y = round(sub['relative_y'].astype(float), 3)

    # # Round all trans coordinates
    # xT = round(trans['relative_x'].astype(float), 3)
    # yT = round(trans['relative_y'].astype(float), 3)

    # plt.scatter(xS, yS, s=20, color='blue')
    # # plt.scatter(x,y, color=colors[i], s=5)
    # plt.scatter(x,y, color='blue', s=5)
    # plt.scatter(xT, yT, color='red', s=5)
    # plt.plot(-3, 8, color='red', marker="o")

    # plt.show()





if __name__ == "__main__":
    main()