import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import ast
import matplotlib.pyplot as plt
import matplotlib
from sklearn.utils import shuffle


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


def renameColVals():
    # Creating the DataFrame
    df = pd.DataFrame({'Date':['10/2/2011', '11/2/2011', '12/2/2011', '13/2/2011'],
                        'Event':['Music', 'Poetry', 'Theatre', 'Comedy'],
                        'Cost':[10000, 5000, 15000, 2000]})
    # Create a dictionary using which we
    # will remap the values
    la = {'Music' : 'M', 'Poetry' : 'P', 'Theatre' : 'T', 'Comedy' : 'C'}
    # Print the dictionary
    print(dict)
    # Remap the values of the dataframe
    df = df.replace({"Event": la})
    # Print the dataframe
    print(df)

def shuf():
    a = [1, 2, 3]
    b = ['a', 'b', 'c']
    a, b = shuffle(np.array(a), np.array(b))
    print(a, b)


def oneHotEncode():
    data =  [
        [['SW'],['SW'], ['SW'], ['SW'], ['SW']],
        [['south'],['south'], ['south'], ['south'], ['south']]
        ]
    ohe = OneHotEncoder()
    a = data[0]
    res = [pd.DataFrame(ohe.fit_transform(x).toarray()) for x in data]
    # res = pd.DataFrame(ohe.fit_transform(data).toarray())
    print(res)


def stepsToOne():
    X = np.array([
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ],
        [
            [1, 9, 2, 8],
            [2, 7, 3, 6]
        ]

        ])
    print(X)
    res = np.concatenate(X, axis=0)
    print(res)


def reshape():
    
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [11, 22, 33]])
    a = X.shape[0] // 2
    print(X.shape)
    X = X.reshape(a, 2, X.shape[1])
    print(X.shape)
    print(X)


def main():

    # reshape()
    # stepsToOne()
    # oneHotEncode()
    # shuf()
    # dupDataframe()
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
    # renameColVals()

    a = [0.0, 0.0, 0.0]
    b = [1.0, 2.0, 3.0]
    c = [4.0, 5.0, 6.0]
    d = [7.0, 8.0, 9.0]
    vals = [a,b,c,d]
    vals = np.array(vals).T
    vals = vals.tolist()
    print(vals)
    res = vals[0] + vals[1] + vals[2]
    res = [[x] for x in res]
    res = np.array(res).T
    print(res)
    print(pd.DataFrame(res))

    mama = pd.DataFrame([[2, 3, 4 ]])
    print("Mama\n", mama)
    sq = mama.squeeze()
    yaya = pd.DataFrame([2,3,4])
    print("Type sq: ", type(sq), " type yaya: ", type(yaya))
    print(sq)
    # vals = [a,b]
    # vals = np.array(vals)
    # print(vals)
    # la = pd.DataFrame(vals)
    # print(la)

if __name__ == "__main__":
    main()