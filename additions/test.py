import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import ast
import matplotlib.pyplot as plt


def copyFirstLines(inFile, outFile, n):
    with open(inFile) as f:
        head = [next(f) for x in range(n)]
    out = open(outFile, "a")

    for i in head:
        out.write(i)
    out.close()
    
def getDirections(fileIn):
    data = pd.read_csv(fileIn, dtype='category')
    queen = data.loc[data.csv_name.str.contains("queen")]
    leith = data.loc[data.csv_name.str.contains('leith')]
    roslyn = data.loc[data.csv_name.str.contains('roslyn')]
    oliver = data.loc[data.csv_name.str.contains('oliver')]
    orchard = data.loc[data.csv_name.str.contains('orchard')]

    names = ["queen", "leith", "roslyn", "oliver", "orchard"]
    sections = [queen, leith, roslyn, oliver, orchard]
    for name, inter in zip(names, sections):
        origin = inter.origin.unique()
        dest = inter.destination.unique()
        print("Directions for ", name, " ", origin.unique(), " ", dest.unique())

def main():

    # # One-hot encoding
    # data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot']
    # values = np.array(data)
    # print(values)
    # le = LabelEncoder()
    # integer_encoded = le.fit_transform(values)
    # print(integer_encoded)
    # ohe = OneHotEncoder(sparse=False)
    # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # onehot_encoded = ohe.fit_transform(integer_encoded)
    # print(onehot_encoded)
    # copyFirstLines("datasets/intersections-dataset-transformed.csv", "datasets/intersections-dataset-transformed-1000.csv", 1000)
    # plt.plot(6, 6, color='blue', marker="o")
    # plt.plot(6, -6, color='blue', marker="o")
    # plt.show()
    # getDirections("datasets/intersections-dataset.csv")
    # data = pd.read_csv("datasets/intersections-dataset.csv", dtype='category')
    # data = data.loc[data.csv_name.str.contains("queen")]
    # origins = data.origin.unique()
    # print("Origins: ", origins)


    a = np.array([1,2]) # [x,y]
    b = np.array([3,5]) # [x,y]

    # Draw a b line.
    (fig, ax) = plt.subplots()
    data_points = np.array([a,b]) # Add points: (1,2) , (3,5)
    print("Data points: \n", data_points)
    test = np.array([
           [-3, 20],
           [7, 10],
           [-3, 0],
           [-13, 10]
        ])
    print()
    print("Test: \n", test)
    data_points_x = data_points[:,0] # For every point, get 1st value, which is x.
    data_points_y = test[:,1] # For every point, get 2nd value, which is y.
    print("Data points x: \n", data_points_x)
    print("Test points y: \n", data_points_y)
    # ax.plot(data_points_x, data_points_y, marker="o", color="k")
    

if __name__ == "__main__":
    main()