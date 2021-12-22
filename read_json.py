import json

import pickle as pkl

import os

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

if __name__ == '__main__':
    path_json = './weather_tokyo.json'

    data = json.load(open(path_json, 'r'))

    data_byday = {}

    for item in data:
        print(item)

        date = item['dt_iso'].split(' ')[0]

        if date not in data_byday:
            data_byday[date] = []

        if 'rain' in item:
            if '1h' in item['rain']:
                rain_1h = item['rain']['1h']
            else:
                rain_1h = -2

            if '3h' in item['rain']:
                rain_3h = item['rain']['3h']
            else:
                rain_3h = -2


            attributes = [item['main']['temp'], item['main']['temp_min'], item['main']['temp_max'], item['main']['feels_like'],
                          item['main']['pressure'], item['main']['humidity'], item['wind']['speed'], item['wind']['deg'],
                          rain_1h, rain_3h, item['clouds']['all'], item['weather'][0]['id']]
        else:
            attributes = [item['main']['temp'], item['main']['temp_min'], item['main']['temp_max'],
                          item['main']['feels_like'],
                          item['main']['pressure'], item['main']['humidity'], item['wind']['speed'],
                          item['wind']['deg'],
                          -1, -1, item['clouds']['all'], item['weather'][0]['id']]
        data_byday[date].append(attributes)
        # print(attributes)

    for date in data_byday:
        # print(date, len(data_byday[date]))
        if len(data_byday[date]) != 24:
            print('error for {}'.format(date))
        # print(data_byday[date])

    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    save_object(data_byday, './outputs/data_byday_20200415.pkl')

