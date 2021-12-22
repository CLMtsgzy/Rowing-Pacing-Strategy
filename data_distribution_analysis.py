import pickle as pkl


def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

import numpy as np

if __name__ == '__main__':
    # data_byday = load_object('./outputs/data_normalized_byday_20200514.pkl')
    data_byday = load_object('./outputs/data_byday_20200415.pkl')


    temp_avghour = np.zeros([24], dtype=np.float64)
    wind_speed = []
    N = len(data_byday)
    for date in data_byday:
        print(data_byday[date])

        for i in range(24):
            temp_avghour[i] += data_byday[date][i][0]
            wind_speed.append(data_byday[date][i][6])

    temp_avghour /= float(N)

    print(temp_avghour)

    print(wind_speed)


    # [0.48659814 0.50454855 0.53090053 0.54564876 0.55421729 0.55903934
    #  0.55670081 0.54897827 0.53093168 0.51377222 0.49599897 0.48646282
    #  0.47929051 0.47313863 0.46053687 0.45331626 0.4476142  0.44093497
    #  0.43598521 0.43081715 0.42607855 0.4253018  0.43905925 0.4684253 ]
