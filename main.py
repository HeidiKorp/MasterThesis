import pandas as pd

from helper import normalizeFeature, normalizeData, balancedClasses, balanceDuplicating, balanceDupHack, getDataById, binarizeFeature, oneHotEncode, codeToDest, codeBinToDest
from model import Model
import numpy as np


def main():
    # TODO run the model
    # Network lengths are: 5, 15 and 25 time-steps.
    # For each length, a new model must be created
    # Correlate to 0.2, 0.6 and 1 seconds in data
    # train=4560, val=1658, test=2074, sum=8292

    # Train percentage: 55%
    # Val percentage: 20%
    # Train percentage: 25%

    # Exactly like paper:
    # Train: 4560
    # Val: 1658
    # Test: 2074

    # Input data should be with uniqueId
    # data = pd.read_csv("../records/records_0-5000.csv", dtype='category')
    # data = pd.read_csv("additions/datasets/intersections-dataset-1000.csv", dtype='category')
    # data = pd.read_csv("additions/datasets/intersections-dataset-transformed-1000.csv", dtype='category')


    # From here normalization
    # data = pd.read_csv("additions/datasets/feb/intersections-dataset-transformed.csv", dtype='category')


    # data = pd.read_csv("additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate-avg2.csv", dtype='category')
    # print("Read the data!")
    # # OneHot encode
    # oneHotEncode(data[['destination']])


    # # # Map destination to a number
    # data = normalizeFeature(data, 'destination', 'code')
    # data['destination'] = data['code'].map(codeToDest)
    # data.to_csv("additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate-avg2.csv")
    # # Map destination to binary
    # data = binarizeFeature(data, 'code', 'codeBin')

    # # uniqueId is the trackId
    # cols = ['relative_x', 'relative_y', 'EgoHeadingRad', 'AbsVelocity', 'uniqueId', 'codeBin']
    # sequence = data[cols]
    # # print("Classes: ", sequence.codeBin.to_string().unique())
    # # Normalize the float values
    # norm = normalizeData(sequence[['relative_x', 'relative_y', 'EgoHeadingRad', 'AbsVelocity']])
    # res = pd.concat([norm, sequence[['uniqueId', 'codeBin']]], ignore_index=True, axis=1)
    
    
    # res.columns = cols
    # print("Res cols:\n", res.columns)
    # # res = balancedClasses(sequence, 'code')
    # # balanceDuplicating(sequence, 'code', "additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate.csv", cols)
    # # balanceDupHack(sequence, 'code', "additions/datasets/feb/ids-to-use2.txt")
    # # getDataById(sequence, "additions/datasets/feb/ids-to-use3.txt", "additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate3.csv")
    # # res.columns = cols
    # res.to_csv("additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate-avg-norm.csv")
    # print("Normalized the data!")


    # Norm2 is with destination
    sequence = pd.read_csv("additions/datasets/feb/intersections-dataset-transformed-balanced-duplicate-avg-norm2.csv", dtype='category')
    sequence = sequence[sequence.columns.drop(list(sequence.filter(regex='Unnamed')))]
    # print("Cols: ", sequence.columns)

    # sequence = None
    # oneHotEncode(sequence[['destination']])
    # Shuffle the dataset
    # shuffled = sequence.sample(frac=1).reset_index()
    # classes = sequence.code.unique()
    # for c in classes:
    #     print("Len for class ", c, " is: ", len(sequence.loc[sequence.code == c]))

    
    # print("All classes: ", sequence.code.unique())
    
    model = Model(sequence,   # This was in
                    0.5,    # dropout
                    512,    # recurrent_width
                    256,    # input_width
                    0.03,   # lreaning_rate
                    0.55,   # train size
                    0.2,    # validation size
                    0.25,   # test size
                    5,      # network length
                    200,
                    "one_layer3",
                    False)    # epochs
    model.set_model() # This was in
    print("Created a model!")

    # # # model.get_model()
    # # # model.compile_model_functional()
    model.train() # This was in
    print("Trained the model!")

    hist = model.get_history()

    best_model = model.get_best_saved_model()
    model.evaluate(best_model, hist)
    print("Evaluated!")

if __name__ == "__main__":
    main()


    # Questions about this model:
    # Is it fully normalized?
    # Check reference 21 - dropout was used on interconnectons only and not on recurrent connections
    # The dataset should be transformed

    # Double check center point in transformTracks.py
    # Double check getThreshold() in transformTracks.py - is it just a guess that's closest to the intersection
    # Make another threshold - per entrance and that is the furthest point of the track with shortest entrance or exit
    
    # The tracks are aligned by distance traveled from the entrance of the intersection
    # For each track, the relative distance traveled before or after crossing the line at the intersection is used
    # As each data sample (track?) contains multiple time steps, the furthest distance in the set is used
    # This results in a fair comparison between RNNs of different lengths

    # Does that mean that they consider all tracks that enter from one intersection and use the closest first location as ref?
