import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle

def oneHotEncode(data):
    ohe = OneHotEncoder()
    res = pd.DataFrame(ohe.fit_transform(data).toarray(), columns=['slow', 'speed', 'wait'])

    # le = LabelEncoder()
    # le.fit_transform(data)
    # encoding = dict(zip(le.classes_, le.transform(le.classes_)))
    # print("Encoding: ", encoding)

    # print(le.classes_)
    return res


def trainSVM(trainX, trainY, testX, testY, saveFolder):
    # https://stackoverflow.com/questions/58074021/training-svc-from-scikit-learn-shows-that-using-h-0-may-be-faster
    # clf = svm.SVC(verbose=True, shrinking=False, kernel='linear').fit(trainX, trainY)
    clf = svm.LinearSVC(verbose=True).fit(trainX, trainY)

    # Save the model
    pickle.dump(clf, open(saveFolder + "svmModel-linear.sav", 'wb'))

    scores = cross_val_score(clf, testX, testY, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    pickle.dump(scores, open(saveFolder + "scores-linear.sav", 'wb'))

def normalizeData(data):
    data = data.astype(float)
    sc = StandardScaler()
    res = sc.fit_transform(data)
    return pd.DataFrame(res, columns=data.columns)

def loadModel(fileName):
    return pickle.load(open(fileName, 'rb'))


def shiftAction(inFile, outFile):
    data = pd.read_csv(inFile, dtype='category')
    ids = data.uniqueId.unique()
    cols = data.columns + "futAction"
    res = pd.DataFrame(columns=cols)
    counter = len(ids)
    for i in ids:
        sub = data.loc[data.uniqueId == i]
        # Shift the actions one step to front
        action = sub.action.to_list()
        # print("** Len of sub: ", len(sub))
        if len(action) > 1:
            action.pop(0)
            action.append(action[-1]) # Repeat the last action
        # print(action)
        sub['futAction'] = action
        # res = res.append(sub)
        counter -= 1
        # res.columns = cols
        sub.to_csv(outFile, mode='a')
        print(counter)


def testData(testFile, exitFile=None):
    test_data = pd.read_csv(testFile, dtype='category')

    if exitFile:
        exitData = pd.read_csv(exitFile, dtype='category')
        


    testY = test_data['action']
    testX = test_data.drop(['action'], axis=1)
    testX = testX[testX.columns.drop(list(testX.filter(regex='Unnamed')))]
    print("Test X cols: ", testX.columns)
    clf = loadModel("svmModel/svmModel.sav")
    print("Loaded model")
    scores = cross_val_score(clf, testX, testY, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    
def main():
    # shiftAction("datasets/feb/training/training-peers-rich.csv", "datasets/feb/training/training-peers-rich-futAction.csv")
    # train_data = pd.read_csv("datasets/feb/training/training-peers-split.csv")
    # validation_data = pd.read_csv("datasets/feb/training/validation-peers-split.csv")
    # test_data = pd.read_csv("datasets/feb/training/test-peers-split.csv")

    train_data = pd.read_csv("datasets/feb/training/training-peers-split-futAction.csv")
    validation_data = pd.read_csv("datasets/feb/training/validation-peers-split-futAction.csv")
    test_data = pd.read_csv("datasets/feb/training/test-peers-split-futAction.csv")

    # train_data = pd.read_csv("datasets/feb/training/training-peers-split-futAction-novel.csv")
    # validation_data = pd.read_csv("datasets/feb/training/validation-peers-split-futAction-novel.csv")
    # test_data = pd.read_csv("datasets/feb/training/test-peers-split-futAction-novel.csv")

    train_data = pd.concat([train_data, validation_data], ignore_index=True)

    # Normalize the data
    # Split into X and y
    # train_y = oneHotEncode(train_data[['action']])
    train_y = train_data['futAction']
    # print("Train cols: ", train_data.columns)
    train_data = train_data.drop(columns='futAction')
    train_data = normalizeData(train_data)
    # print(train_data.head())

    # test_y = oneHotEncode(test_data[['action']])
    test_y = test_data['futAction']
    test_data = test_data.drop(columns='futAction')
    test_data = normalizeData(test_data)

    # Normalize train and test X

    trainSVM(train_data, train_y, test_data, test_y, "svmModel2-re2/")
    # test_data = pd.read_csv("datasets/april/testFiles/SVMtest_orchard.csv")


    


if __name__ == "__main__":
    main()