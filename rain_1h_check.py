import pickle as pkl

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

if __name__ == '__main__':
    # path_data = './outputs/data_byday_20200415.pkl'
    # data_byday = load_object(path_data)
    #
    #
    # rain1h_dict = {}
    #
    # for date in data_byday:
    #     data = data_byday[date]
    #     for item in data:
    #         rain1h, id = item[8], item[11]
    #
    #         if rain1h >= 0:
    #             if id not in rain1h_dict:
    #                 rain1h_dict[id] = []
    #
    #             rain1h_dict[id].append(float(rain1h))
    #
    # print(rain1h_dict)
    #
    # for key in rain1h_dict:
    #     print('{}: {}, len = {}'.format(key, sum(rain1h_dict[key]) / float(len(rain1h_dict[key])), len(rain1h_dict[key])))
    #

    data_min = load_object('./outputs/data_min_20200514.pkl')
    data_max = load_object('./outputs/data_max_20200514.pkl')
    print(data_min)
    print(data_max)