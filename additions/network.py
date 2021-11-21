import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize, Normalizer
import ast



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

    # print("Type: ", type(validation_data['PeerX']))

    # train_Y = train_data.action
    # test_Y = test_data.action
    val_Y = validation_data.action
    # print("Before norm: \n", validation_data.head(2))

    # train_X = train_data[cols]
    # test_X = test_data[cols]
    
    val_main = validation_data[main_cols]
    val_peers = validation_data[peer_cols]

    # Normalize the data
    sc = StandardScaler()
    val_main = sc.fit_transform(val_main)

    for col in val_peers.columns:
        # [[] if x=='[]' else x for x in val_peers[col]]
        a = val_peers[col]
        # print("Start\n", a.head(2))
        # print()

        data = a.to_numpy()
        
       
        data = [x for x in data if x != '[]']
        print(type(data[0]))
        data = ast.literal_eval(data)
        # print("Row 1: ", data[0])
        # for i in rows:
        #     for j in i:
        #         j = float(j)
        # print("row 1: ", rows[0])
        # df = pd.DataFrame(rows).astype(float)
        # print(df.head(2))
        # df = pd.DataFrame(val_peers[col], columns = peer_cols)
        # df = val_peers[col].to_frame()

        
        # print(df.head(2))

        # scaler = sc.fit(df)
        # scaled = scaler.transform(df)
        # arr = scaled.to_numpy()
        # print(arr[:2])

        # next = 0
        # for i in range(len(a)):
        #     if a[i] != '[]':
        #         a[i] = arr[next]
        #         next += 1



        # val_peers[col] = scaled.squeeze()
        # print(val_peers[col].head(5))
        # print()


        # transformer = Normalizer().fit(df)

    #     a = val_peers[col]
    #     print("A: ", a)
    #     print("Type: ", type(a))

    #     transformer = Normalizer().fit(a)
    #     a = transformer.transform(a)
    #     val_peers[col] = a
    #     # print("A type: ", type(a))
    #     print("A: ", a)
    #     # a = normalize(a)
    #     # val_peers[col] = sc.fit_transform(val_peers[col])
    
    # val_X = val_main + val_peers
    # print("After norm: \n", val_X.head(2))


if __name__ == "__main__":
    main()