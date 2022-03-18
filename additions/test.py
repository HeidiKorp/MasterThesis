import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import ast
import matplotlib.pyplot as plt
import matplotlib


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

def aboveOrBelow():
    is_above = lambda p,a,b: np.cross(p-a, b-a) < 0
    
    # Data.
    # a = np.array([1,2]) # [x,y]
    # b = np.array([3,5]) # [x,y]
    a = np.array([3,1]) # [x,y]
    b = np.array([3,5]) # [x,y]
    
    # p1 = np.array([2,4]) # [x,y]
    # p2 = np.array([2,3]) # [x,y]
    p1 = np.array([2,4]) # [x,y]
    p2 = np.array([4,4]) # [x,y]
    
    
    # Draw a b line.
    (fig, ax) = plt.subplots()
    data_points = np.array([a,b]) # Add points: (1,2) , (3,5)
    data_points_x = data_points[:,0] # For every point, get 1st value, which is x.
    data_points_y = data_points[:,1] # For every point, get 2nd value, which is y.
    ax.plot(data_points_x, data_points_y, marker="o", color="k")
    
    # Draw point: color point if it is above or below line.
    # Point 1:
    if is_above(p1,a,b):
        ax.scatter(p1[0], p1[1], color='green')
    else:
        ax.scatter(p1[0], p1[1], color='red')
    
    # Point 2:
    if is_above(p2,a,b):
        ax.scatter(p2[0], p2[1], color='green')
    else:
        ax.scatter(p2[0], p2[1], color='red')
    plt.show()


def datasetToDict():
    data = [['leith', 'north', 5], ['leith', 'south', 6], ['roslyn', 'east', 7], ['roslyn', 'west', 8]]
    df = pd.DataFrame(data, columns=['name', 'origin', 'dist'])

    names = list(df.name)
    origs = list(df.origin)
    tuples = zip(names, origs)
    print(tuples)

    new_list = []
    for i in tuples:
        a = i[0] + "-" + i[1]
        new_list.append(a)
        print(a)

    d = dict(zip(new_list, list(df.dist)))
    print("D: \n", d)
    # d = {}
    # d['name'] = list(df.name)
    # print("D: \n", d)
    # for name in d['name']:
    #     print("Name is: ". name)
    #     sub1 = df.loc[df.name == name]
    #     d['name'][name]['origin'] = sub1.origin
    #     print("D2: ", d)
    #     for ori in d['name'][name]['origin'].keys():
    #         sub2 = sub1.loc[sub1.origin == ori]
    #         d['name'][name]['origin'][ori]['dist'] = sub2.dist
    # print("D3: ", d)

def dupDataframe():
    data = pd.DataFrame()
    data['id'] = [1, 2, 3]
    data['time'] = [22, 33, 44]
    print("Orig data: \n", data)
    newdf = pd.DataFrame(np.repeat(data.values, 3, axis=0))
    newdf.columns = data.columns
    print(newdf)
    
def main():

    dupDataframe()
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

    # aboveOrBelow()

    # ax.plot(data_points_x, data_points_y, marker="o", color="k")
    # datasetToDict()

if __name__ == "__main__":
    main()