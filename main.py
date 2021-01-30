import pandas as pd

from .helper import normalizeFeature
from .model import Model


def main():
    # TODO run the model
    # Network lengths are: 5, 15 and 25 time-steps.
    # For each length, a new model must be created
    # Correlate to 0.2, 0.6 and 1 seconds in data
    # train=4560, val=1658, test=2074, sum=8292. Why?
    # train+val = 6218 ~ 75%

    # Train percentage: 55%
    # Val percentage: 20%
    # Train percentage: 25%

    # Exactly like paper:
    # Train: 4560
    # Val: 1658
    # Test: 2074

    # Input data should be with objectId
    data = pd.read_csv("../records/records_0-5000.csv", dtype='category')
    X = data[['relative_x', 'relative_y', 'EgoHeadingRad', 'AbsVelocity', 'ObjectId']]

    mapping = normalizeFeature(data, 'destination', 'code')
    print(mapping)


    y = data['code']
    model = Model(X.head(),        # X
                    y.head(),      # y
                    0.5,    # dropout
                    512,    # recurrent_width
                    256,    # input_width
                    0.03,   # lreaning_rate
                    0.55,   # train size
                    0.2,    # validation size
                    0.25,   # test size
                    5,      # network length
                    20)     # epochs
    # model.getModel()
    # model.setModel()
    # model.train()

if __name__ == "__main__":
    main()