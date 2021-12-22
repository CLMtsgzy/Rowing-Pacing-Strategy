import pickle as pkl
import numpy as np

from math import pi

from data_normalized import weatherid2rain



def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

tab_predict_normal_paras = [2., 2., 0.05] # temp_min, temp_max, rain_prob
cn_predict_normal_paras = [1., 1., 0.03, 0.5, 0.5] # temp_min, temp_max, rain_prob, wind_x, wind_y

TEMP_MIN_LC = -11.05
TEMP_MIN_HC = 34.6
TEMP_MAX_LC = -2.24
TEMP_MAX_HC = 43.89
WIND_SPEED_X_LC = -33.97586748
WIND_SPEED_X_HC = 24.94153163
WIND_SPEED_Y_LC = -23.21040773
WIND_SPEED_Y_HC = 19.5
WIND_SPEED_LC = 0.
WIND_SPEED_HC = 34.5

def normalize(x, lc, hc):
    return (x - lc) / (hc - lc)

np.random.seed(1989)

if __name__ == '__main__':
    data_normalized_byday = load_object('./outputs/data_normalized_byday_20200514.pkl')
    data_min = load_object('./outputs/data_min_20200514.pkl')
    data_max = load_object('./outputs/data_max_20200514.pkl')
    data_byday = load_object('./outputs/data_byday_20200415.pkl')

    date_list = []
    temp_min_list = []
    temp_max_list = []
    rain_total_list = []
    rain_prob_list = []
    wind_x_list = []
    wind_y_list = []
    wind_speed_list = []

    tab_predict_temp_min_list = []
    tab_predict_temp_max_list = []
    tab_predict_rain_prob_list = []

    cn_predict_temp_min_day_list = []
    cn_predict_temp_min_night_list = []
    cn_predict_temp_max_day_list = []
    cn_predict_temp_max_night_list = []
    cn_predict_rain_prob_day_list = []
    cn_predict_rain_prob_night_list = []
    cn_predict_wind_x_day_list = []
    cn_predict_wind_x_night_list = []
    cn_predict_wind_y_day_list = []
    cn_predict_wind_y_night_list = []


    for date in data_byday:
        date_list.append(date)
        data_thisday = np.array(data_byday[date])

        temp_min = np.min(data_thisday[:, 1])
        temp_min_night = np.min(data_thisday[9:21, 1])
        temp_min_day = min(np.min(data_thisday[:9, 1]), np.min(data_thisday[21:, 1]))
        temp_max = np.max(data_thisday[:, 2])
        temp_max_night = np.max(data_thisday[9:21, 2])
        temp_max_day = max(np.max(data_thisday[:9, 2]), np.max(data_thisday[21:, 2]))

        rains_list = []
        for i in range(len(data_thisday)):
            item = data_thisday[i, :]
            if item[8] >= 0.:
                rains_list.append(item[8])
            elif item[9] >= 0.:
                rains_list.append(item[9] / 3.)
            else:
                rains_list.append(weatherid2rain[item[11]])

        rain_total = sum(rains_list)
        rain_prob = rain_total / 5.

        rain_prob_night = sum(rains_list[9:21]) / 2.5
        rain_prob_day = (sum(rains_list[:9]) + sum(rains_list[21:])) / 2.5

        wind_xs = np.cos(data_thisday[:, 7] * pi / 180.) * data_thisday[:, 6]
        wind_ys = np.sin(data_thisday[:, 7] * pi / 180.) * data_thisday[:, 6]

        for item in data_thisday[:, 6]:
            wind_speed_list.append(item)


        wind_avgx = np.mean(wind_xs)
        wind_avgy = np.mean(wind_ys)

        wind_avgx_night = np.mean(wind_xs[9:21])
        wind_avgx_day = (np.sum(wind_xs[:9]) + np.sum(wind_xs[21:])) / 12.
        wind_avgy_night = np.mean(wind_ys[9:21])
        wind_avgy_day = (np.sum(wind_ys[:9]) + np.sum(wind_ys[21:])) / 12.


        # print(data_thisday[:, 6])
        # print('date = {}, windspeed = {}, wind = ({}, {})'.format(date, wind_speed, wind_avgx, wind_avgy))
        # print()


        temp_min_list.append(temp_min)
        temp_max_list.append(temp_max)
        rain_total_list.append(rain_total)
        rain_prob_list.append(rain_prob)
        wind_x_list.append(wind_avgx)
        wind_y_list.append(wind_avgy)


        temp_min_predict = temp_min + np.random.randn() * tab_predict_normal_paras[0]
        temp_max_predict = temp_max + np.random.randn() * tab_predict_normal_paras[1]
        rain_prob_predict = rain_prob + np.random.randn() * tab_predict_normal_paras[2]
        rain_prob_predict = max(rain_prob_predict, 0)
        rain_prob_predict = min(rain_prob_predict, 1)

        tab_predict_temp_min_list.append(temp_min_predict)
        tab_predict_temp_max_list.append(temp_max_predict)
        tab_predict_rain_prob_list.append(rain_prob_predict)


        temp_min_night_predict = temp_min_night + np.random.rand() * cn_predict_normal_paras[0]
        temp_min_day_predict = temp_min_day + np.random.rand() * cn_predict_normal_paras[0]
        temp_max_night_predict = temp_max_night + np.random.rand() * cn_predict_normal_paras[1]
        temp_max_day_predict = temp_max_day + np.random.rand() * cn_predict_normal_paras[1]
        rain_prob_night_predict = rain_prob_night + np.random.randn() * tab_predict_normal_paras[2]
        rain_prob_day_predict = rain_prob_day + np.random.randn() * tab_predict_normal_paras[2]
        rain_prob_night_predict = min(max(rain_prob_night_predict, 0), 1)
        rain_prob_day_predict = min(max(rain_prob_day_predict, 0), 1)
        wind_x_night_predict = wind_avgx_night + np.random.rand() * cn_predict_normal_paras[3]
        wind_x_day_predict = wind_avgx_day + np.random.rand() * cn_predict_normal_paras[3]
        wind_y_night_predict = wind_avgy_night + np.random.rand() * cn_predict_normal_paras[4]
        wind_y_day_predict = wind_avgy_day + np.random.rand() * cn_predict_normal_paras[4]


        cn_predict_temp_min_day_list.append(temp_min_day_predict)
        cn_predict_temp_min_night_list.append(temp_min_night_predict)
        cn_predict_temp_max_day_list.append(temp_max_day_predict)
        cn_predict_temp_max_night_list.append(temp_max_night_predict)
        cn_predict_rain_prob_day_list.append(rain_prob_day_predict)
        cn_predict_rain_prob_night_list.append(rain_prob_night_predict)
        cn_predict_wind_x_day_list.append(wind_x_day_predict)
        cn_predict_wind_x_night_list.append(wind_x_night_predict)
        cn_predict_wind_y_day_list.append(wind_y_day_predict)
        cn_predict_wind_y_night_list.append(wind_y_night_predict)






    # rain_total_list.sort()
    # print(rain_total_list)

    gt_dict = {}

    for date in data_byday:
        data_thisday = np.array(data_byday[date])

        temp_min_thisday = data_thisday[:, 1]
        temp_max_thisday = data_thisday[:, 2]

        wind_xs_thisday = np.cos(data_thisday[:, 7] * pi / 180.) * data_thisday[:, 6]
        wind_ys_thisday = np.sin(data_thisday[:, 7] * pi / 180.) * data_thisday[:, 6]

        wind_speed_thisday = data_thisday[:, 6]

        rains_list_thisday = []
        for i in range(len(data_thisday)):
            item = data_thisday[i, :]
            if item[8] >= 0.:
                rains_list_thisday.append(item[8])
            elif item[9] >= 0.:
                rains_list_thisday.append(item[9] / 3.)
            else:
                rains_list_thisday.append(weatherid2rain[item[11]])

        gt_thisday = []

        for j in range(len(temp_min_thisday)):
            gt_thisday.append(normalize(temp_min_thisday[j], TEMP_MIN_LC, TEMP_MIN_HC))
            gt_thisday.append(normalize(temp_max_thisday[j], TEMP_MAX_LC, TEMP_MAX_HC))
            rain_prob = max(min(rains_list_thisday[j] / 0.10, 1), 0)
            gt_thisday.append(rain_prob)
            gt_thisday.append(normalize(wind_xs_thisday[j], WIND_SPEED_X_LC, WIND_SPEED_X_HC))
            gt_thisday.append(normalize(wind_ys_thisday[j], WIND_SPEED_Y_LC, WIND_SPEED_Y_HC))
            gt_thisday.append(normalize(wind_speed_thisday[j], WIND_SPEED_LC, WIND_SPEED_HC))

        gt_dict[date] = gt_thisday







    tab_predict_dict = {}
    for i in range(len(date_list)):
        date = date_list[i]
        tab_predict_temp_min = normalize(tab_predict_temp_min_list[i], TEMP_MIN_LC, TEMP_MIN_HC)
        tab_predict_temp_max = normalize(tab_predict_temp_max_list[i], TEMP_MAX_LC, TEMP_MAX_HC)
        tab_predict_rain_prob = tab_predict_rain_prob_list[i]

        tab_predict_dict[date] = [tab_predict_temp_min, tab_predict_temp_max, tab_predict_rain_prob]


    cn_predict_dict = {}
    for i in range(len(date_list)):
        date = date_list[i]
        cn_predict_temp_min_day = normalize(cn_predict_temp_min_day_list[i], TEMP_MIN_LC, TEMP_MIN_HC)
        cn_predict_temp_min_night = normalize(cn_predict_temp_min_night_list[i], TEMP_MIN_LC, TEMP_MIN_HC)
        cn_predict_temp_max_day = normalize(cn_predict_temp_max_day_list[i], TEMP_MAX_LC, TEMP_MAX_HC)
        cn_predict_temp_max_night = normalize(cn_predict_temp_max_night_list[i], TEMP_MAX_LC, TEMP_MAX_HC)
        cn_predict_rain_prob_day = cn_predict_rain_prob_day_list[i]
        cn_predict_rain_prob_night = cn_predict_rain_prob_night_list[i]
        cn_predict_wind_x_day = normalize(cn_predict_wind_x_day_list[i], WIND_SPEED_X_LC, WIND_SPEED_X_HC)
        cn_predict_wind_x_night = normalize(cn_predict_wind_x_night_list[i], WIND_SPEED_X_LC, WIND_SPEED_X_HC)
        cn_predict_wind_y_day = normalize(cn_predict_wind_y_day_list[i], WIND_SPEED_Y_LC, WIND_SPEED_Y_HC)
        cn_predict_wind_y_night = normalize(cn_predict_wind_y_night_list[i], WIND_SPEED_Y_LC, WIND_SPEED_Y_HC)

        cn_predict_dict[date] = [cn_predict_temp_min_day, cn_predict_temp_min_night, cn_predict_temp_max_day, cn_predict_temp_max_night,
                                 cn_predict_rain_prob_day, cn_predict_rain_prob_night, cn_predict_wind_x_day, cn_predict_wind_x_night,
                                 cn_predict_wind_y_day, cn_predict_wind_y_night]



    print(len(wind_speed_list))

    print('temp_min = ({}, {})'.format(min(temp_min_list), max(temp_min_list)))
    print('temp_max = ({}, {})'.format(min(temp_max_list), max(temp_max_list)))
    print('wind_speed = ({}, {}).'.format(min(wind_speed_list), max(wind_speed_list)))


    # print(gt_dict)
    #
    save_object(gt_dict, './outputs/gt_dict_20200524.pkl')
    save_object(tab_predict_dict, './outputs/tab_predict_dict_20200528.pkl')
    save_object(cn_predict_dict, './outputs/cn_predict_dict_20200528.pkl')