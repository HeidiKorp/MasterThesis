import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize, Normalizer


def main():
    my_array = np.array([[11,22,33],[44,55,66], NaN])

    df = pd.DataFrame(my_array)
    print(df)

    sc = StandardScaler()
    val_main = sc.fit_transform(df)
    print(val_main)

if __name__ == "__main__":
    main()