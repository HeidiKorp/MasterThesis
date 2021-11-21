# Next:
# Get random 68921 rows from not-waiting-rich.csv file and save them into another file
# Add a new column to both datasets: 0 for waiting and 1 for not waiting
# Add the two datasets and shuffle rows
# Create a model that gets one snapshot fed into it and it predicts 0 or 1 (wait or not)
import pandas as pd
import numpy as np


def getRandomRows(dataFile, n, outFile):
    data = pd.read_csv(dataFile, dtype='category')
    sub = data.sample(n)
    sub.to_csv(outFile)


def deleteExcessColumns(fileName, start, end, outFile):
    data = pd.read_csv(fileName, dtype='category')
    cols = len(data.columns)

    new_data = data.iloc[:, start:cols+end]
    new_data.to_csv(outFile)


def deleteUnnamedCols(inFile, outFile):
    data = pd.read_csv(inFile, dtype='category')
    cols = [c for c in data.columns if 'unnamed' not in c.lower()]
    sub = data[cols]
    sub.to_csv(outFile)


def addWaitingIndicatorColumn(dataFile, indicator, outFile):
    data = pd.read_csv(dataFile, dtype='category')
    data['waiting'] = indicator
    data.to_csv(outFile)


def concatDataFiles(f1, f2, outFile):
    d1 = pd.read_csv(f1, dtype='category')
    d2 = pd.read_csv(f2, dtype='category')
    frames = [d1, d2]
    res = pd.concat(frames)
    res = res.sample(frac=1)
    res.to_csv(outFile)



def main():
    # getRandomRows("../datasets/not-waiting-rich-not-empty.csv", 68921, '../datasets/not-waiting-rich-not-empty-small.csv')
    # deleteExcessColumns("../datasets/not-waiting-rich-small.csv", 3, 0, "../datasets/not-waiting-rich-small2.csv")
    deleteUnnamedCols("../datasets/full-rich.csv", "../datasets/full-rich.csv")
    # addWaitingIndicatorColumn("../datasets/not-waiting-rich-not-empty-small.csv", 1, "../datasets/not-waiting-rich-not-empty-small-indicator.csv")
    # concatDataFiles("../datasets/not-waiting-rich-not-empty-small-indicator.csv", "../datasets/waiting-rich-not-empty-indicator.csv", "../datasets/full-rich.csv")


if __name__ == '__main__':
    main()