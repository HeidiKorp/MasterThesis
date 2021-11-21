import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize, Normalizer
import ast


def normalizeCol(col_data, sc):
    orig_data = col_data.to_numpy()
    
    # Remove empty rows
    data = [x for x in orig_data if x != '[]']

    # Convert data to list of floats
    for i in range(len(data)):
        data[i] = ast.literal_eval(data[i])
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
            
    df = pd.DataFrame(data)
    scaled = sc.fit_transform(df)

    # Add back empty rows
    res = []
    next = 0
    for i in range(len(orig_data)):
        if orig_data[i] == '[]':
            res.append([])
        else:
            res.append(scaled[next])
            next += 1
    
    return res


def main():
    # train_data = pd.read_csv("datasets/training/training-peers-rich.csv", dtype='category')
    # test_data = pd.read_csv("datasets/training/test-peers-rich.csv", dtype='category')
    validation_data = pd.read_csv("datasets/training/validation-peers-rich.csv", dtype='category')

    main_cols = ['relative_x', 
            'relative_y', 
            'RelVelocity_X', 
            'RelVelocity_Y',
            'AbsVelocity_X',
            'AbsVelocity_Y',
            'EgoHeadingRad'
    ]
    peer_cols = ['PeerX',
            'PeerY',
            'PeerRelVelX',
            'PeerRelVelY',
            'PeerAbsVelX',
            ]

    val_Y = validation_data.action
    
    val_main = validation_data[main_cols]
    val_peers = validation_data[peer_cols]

    # Normalize the data
    sc = StandardScaler()
    val_main = sc.fit_transform(val_main)
    val_main = pd.DataFrame(val_main, columns=main_cols)

    norm_peer = pd.DataFrame(columns=peer_cols)
    for col in peer_cols:
        norm_peer[col] = normalizeCol(val_peers[col], sc)
    
    val_X = pd.concat([val_main, norm_peer])
    print("After norm: \n", val_X.head(2))


if __name__ == "__main__":
    main()