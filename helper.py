from sklearn.preprocessing import LabelEncoder

def normalizeFeature(data, feature, newCol):
    le = LabelEncoder()
    data[newCol] = le.fit_transform(data[feature])
    return dict(zip(le.classes_, le.transform(le.classes_)))

def unNormalize(mapping, res):
    for dest, code in mapping:
        if code == res:
            return dest


def pad(block, n):
    for k in range(n):
        block.append(block[-1])
    return block

    
def blockify(arr, n):
    counter = 0
    res = []
    prev_id = -1

    for i in range(len(arr)):
        if prev_id == arr[i] and counter < n:
            res[-1].append(arr[i]) 
            counter += 1
        else:
            if prev_id != arr[i] and counter > 0:
                pad(res[-1], n-counter)
            res.append([arr[i]])
            counter = 1
            prev_id = arr[i]

        if i == len(arr) - 1 and counter < n:  # If it it the last block and there are less than 5 records
            pad(res[-1], n-counter)
    return res