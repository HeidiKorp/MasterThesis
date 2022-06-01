import matplotlib.pyplot as plt
import math
import pandas as pd

def rotate(x, y, rad):
    res_x = math.cos(rad) * x - math.sin(rad) * y
    res_y = math.sin(rad) * x + math.cos(rad) * y
    return res_x, res_y

def plotShape():
    # x = [0, 5, 6, 5, 0]
    # y = [0, 0, 1.5, 3, 3]
    data = pd.read_csv("datasets/testing/leith.csv", dtype='category')
    x = data['relative_x_trans'].astype(float)
    y = data['relative_y_trans'].astype(float)


    # x = [5, 4, 3, 4, 5, 10, 11, 12, 11, 10]
    # y = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

    res_x = []
    res_y = []
    for i, j in zip(x, y):
        rx, ry = rotate(i, j, math.radians(15))
        res_x.append(rx)
        res_y.append(ry)
    plt.scatter(x, y, s=1)
    # plt.scatter(res_x, res_y, s=1)
    plt.axis('equal')
    plt.axis([-20, 20, -20, 40])
    plt.show()


plotShape()