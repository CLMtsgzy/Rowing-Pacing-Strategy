import pickle as pkl

import numpy as np
from math import pi

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

# weatherid2rain = {200: 0.21, 201: 0.73, 202: 1.56, 210: 0.21, 211: 0.73, 212: 1.56, 221: 0.73, 230: 0.21,
#                   231: 0.73,
#                   232: 1.56, 300: 0.10, 301: 0.21, 302: 0.31, 310: 0.10, 311: 0.21, 312: 0.31, 313: 0.21,
#                   314: 0.31,
#                   321: 0.21, 500: 0.21, 501: 0.73, 502: 1.56, 503: 3.13, 504: 7.29, 511: 0.73, 520: 0.21,
#                   521: 0.73,
#                   522: 1.56, 531: 0.73, 600: 0.05, 601: 0.16, 602: 0.31, 611: 0.16, 612: 0.05, 613: 0.16,
#                   615: 0.13,
#                   616: 0.44, 620: 0.05, 621: 0.16, 622: 0.31, 701: 0.00, 711: 0.00, 721: 0.00, 731: 0.00,
#                   741: 0.00,
#                   751: 0.00, 761: 0.00, 762: 0.00, 771: 0.00, 781: 0.00, 800: 0.00, 801: 0.00, 802: 0.00,
#                   803: 0.00,
#                   804: 0.00}

weatherid2rain = {200: 3.12, 201: 17.09, 202: 42.00, 210: 3.12, 211: 17.09, 212: 42.00, 221: 17.09, 230: 3.12,
                  231: 17.09,
                  232: 42.00, 300: 0.20, 301: 0.42, 302: 0.62, 310: 0.20, 311: 0.42, 312: 0.62, 313: 0.42,
                  314: 0.62,
                  321: 0.42, 500: 0.38, 501: 1.87, 502: 6.57, 503: 23.28, 504: 60.00, 511: 0.73, 520: 5.61,
                  521: 14.29,
                  522: 29.38, 531: 5.61, 600: 0.10, 601: 0.32, 602: 0.32, 611: 0.32, 612: 0.10, 613: 0.32,
                  615: 0.26,
                  616: 0.88, 620: 0.10, 621: 0.32, 622: 0.62, 701: 0.00, 711: 0.00, 721: 0.00, 731: 0.00,
                  741: 0.00,
                  751: 0.00, 761: 0.00, 762: 0.00, 771: 0.00, 781: 0.00, 800: 0.00, 801: 0.00, 802: 0.00,
                  803: 0.00,
                  804: 0.00}

if __name__ == '__main__':
    path_data = './outputs/data_byday_20200415.pkl'



    data_byday = load_object(path_data)

    attributes_num = len(data_byday['2020-01-07'][0])
    print('attributes_num = {}'.format(attributes_num))

    data_total = []

    # index: 6, 7, decompose it to W-E, N-S dir speed
    for date in data_byday:
        for item in data_byday[date]:
            item_copy = np.copy(item)
            speed_WE = np.cos(item[7] * pi / 180.) * item[6]
            speed_NS = np.sin(item[7] * pi / 180.) * item[6]
            item_copy[6], item_copy[7] = speed_WE, speed_NS
            data_total.append(item_copy)

    print(len(data_total))

    data_total = np.array(data_total)
    data_min = np.min(data_total, axis=0)
    data_max = np.max(data_total, axis=0)
    print(data_min, data_max)

    cnt_rain = 0
    data_normalized_byday = {}
    for date in data_byday:
        data_thisday = []
        for item in data_byday[date]:
            data_thisitem = []
            for i in range(8):
                data_thisitem.append((item[i] - data_min[i]) / (data_max[i] - data_min[i]))

            data_thisitem.append((item[10] - data_min[10]) / (data_max[10] - data_min[10]))

            if item[8] >= 0.:
                data_thisitem.append(item[8])
            elif item[9] >= 0.:
                data_thisitem.append(item[9] / 3.)
            else:
                data_thisitem.append(weatherid2rain[item[11]])

            if item[8] >= 0. or item[8] == -2:
                cnt_rain += 1



            data_thisday.append(data_thisitem)

        data_normalized_byday[date] = data_thisday

    # print(data_normalized_byday)

    print(cnt_rain)
    print(len(data_byday))

    save_object(data_normalized_byday, './outputs/data_normalized_byday_20200514.pkl')
    save_object(data_min, './outputs/data_min_20200514.pkl')
    save_object(data_max, './outputs/data_max_20200514.pkl')