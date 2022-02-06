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
    plt.plot(6, 6, color='blue', marker="o")
    plt.plot(6, -6, color='blue', marker="o")
    plt.show()
    

if __name__ == "__main__":
    main()