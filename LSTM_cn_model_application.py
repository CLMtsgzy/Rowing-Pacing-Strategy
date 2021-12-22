from LSTM_cn_20200524 import LSTM
import torch

import numpy as np

import pickle as pkl

import generate_gt_predict

from math import sqrt, pi, sin, cos



degrees_dict = {'北':0, '东北偏北': 22.5, '东北': 45, '东北偏东': 67.5, '东': 90, '东南偏东': 112.5, '东南': 135, '东南偏南': 157.5,
                '南': 180, '西南偏南': 202.5, '西南': 225, '西南偏西': 247.5, '西': 270, '西北偏西': 292.5, '西北': 315, '西北偏北': 337.5}


windspeeds_dict = {0: 0.1, 1: 0.9, 2: 2.45, 3: 4.4, 4: 6.8, 5: 9.35, 6: 12.3, 7: 15.5, 8: 18.95, 9: 22.6, 10: 26.45,
                   11: 29.55}

def getCnPredict():
    cn_predict_dict = {}
    cn_day_predict_dict = {}

    with open('./crawled/weathercn_20200414', 'r') as f:
        head = f.readline()
        while True:
            line = f.readline()
            if line:
                items = line.split('\t')
                date = items[0]
                period = items[1]
                high_temp, low_temp, rain_prob = items[2], items[3], items[4]
                high_temp, low_temp, rain_prob = float(high_temp), float(low_temp), float(rain_prob)
                wind_low, wind_high = items[5], items[6]
                direction = wind_low.split('风')[0]
                level_low = int(wind_low.split('风')[1].split('级')[0])
                level_high = int(wind_high.split('风')[1].split('级')[0])
                wind_degree = degrees_dict[direction]
                windspeed_avg = (windspeeds_dict[level_low] + windspeeds_dict[level_high]) / 2.
                windx = windspeed_avg * cos(wind_degree * pi / 180.)
                windy = windspeed_avg * sin(wind_degree * pi / 180.)

                high_temp = generate_gt_predict.normalize(high_temp, generate_gt_predict.TEMP_MAX_LC, generate_gt_predict.TEMP_MAX_HC)
                low_temp = generate_gt_predict.normalize(low_temp, generate_gt_predict.TEMP_MIN_LC, generate_gt_predict.TEMP_MIN_HC)
                windx = generate_gt_predict.normalize(windx, generate_gt_predict.WIND_SPEED_X_LC, generate_gt_predict.WIND_SPEED_X_HC)
                windy = generate_gt_predict.normalize(windy, generate_gt_predict.WIND_SPEED_Y_LC, generate_gt_predict.WIND_SPEED_Y_HC)

                if period == 'day':
                    cn_day_predict_dict[date] = [low_temp, high_temp, rain_prob, windx, windy]
                else:
                    if date in cn_day_predict_dict:
                        low_temp_day, high_temp_day, rain_prob_day, windx_day, windy_day = cn_day_predict_dict[date]
                        cn_predict_dict[date] = [low_temp_day, low_temp, high_temp_day, high_temp, rain_prob_day, rain_prob,
                                                 windx_day, windx, windy_day, windy]

            else:
                break
    return cn_predict_dict


def reverseNormalize(x, lc, hc):
    return x * (hc - lc) + lc

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)


def output2result(output, feature_num_perhour=6):
    m, n = len(output), len(output[0])
    result_list = np.zeros([m, n*5//6], dtype=np.float64)

    for i in range(m):
        x_side, y_side = 0., 0.
        j_out = 0
        for j in range(n):
            if j % feature_num_perhour == 0:
                result_list[i, j_out] = reverseNormalize(output[i, j], generate_gt_predict.TEMP_MIN_LC, generate_gt_predict.TEMP_MIN_HC)
                j_out += 1
            elif j % feature_num_perhour == 1:
                result_list[i, j_out] = reverseNormalize(output[i, j], generate_gt_predict.TEMP_MAX_LC, generate_gt_predict.TEMP_MAX_HC)
                j_out += 1
            elif j % feature_num_perhour == 2:
                result_list[i, j_out] = output[i, j]
                j_out += 1
            elif j % feature_num_perhour == 3:
                x_side = output[i, j]
            elif j % feature_num_perhour == 4:
                y_side = output[i, j]
            elif j % feature_num_perhour == 5:
                norm_factor = sqrt(x_side * x_side + y_side * y_side)
                wind_speed = reverseNormalize(output[i, j], generate_gt_predict.WIND_SPEED_LC, generate_gt_predict.WIND_SPEED_HC)
                wind_x = wind_speed * x_side / norm_factor
                wind_y = wind_speed * y_side / norm_factor
                result_list[i, j_out] = wind_x
                j_out += 1
                result_list[i, j_out] = wind_y
                j_out += 1

    return result_list


def aggregateResultList(result_list):
    print(result_list[:, -1])

    result_agg = np.median(result_list, axis=0)
    print(result_agg)


    for i in range(len(result_agg) // 5):
        print('{}h, result = {}'.format(i, result_agg[i*5:(i+1)*5]))




if __name__ == '__main__':
    path_model = './models/cn_lstm_epoch19_20200524.pkl'

    training_days = load_object('./outputs/training_days.pkl')
    data_byday = load_object('./outputs/data_normalized_byday_20200514.pkl')




    print(data_byday['2010-08-01'])


    seq_num = 31*24
    feature_num = 10
    flat_feature_num = 10
    output_num = 6*24

    lstm = LSTM(hidden_size=64, feature_num=feature_num, flat_feature_num=flat_feature_num, output_num=output_num)
    lstm.load_state_dict(torch.load(path_model, map_location=lambda storage, loc: storage))

    print(lstm)

    output_day = '2020-06-21'
    date_noyear = output_day[5:]
    # print(date_noyear)

    train_thisday = training_days[date_noyear]

    training_samples = []

    for i in range(len(train_thisday)):
        if train_thisday[i][-1] < output_day:
            # print(train_thisday[i][-1], start_day)
            training_samples.append([output_day, i])

    N = len(training_samples)

    data_x1s = np.zeros([N, seq_num, feature_num], dtype=np.double)
    data_x2 = np.zeros([N, flat_feature_num], dtype=np.double)
    label_gt = np.zeros([N, output_num], dtype=np.double)
    for k in range(N):
        date, idx = training_samples[k]
        training_day = training_days[date[5:]][idx]
        # print(training_day)
        for l in range(len(training_day)):
            day = training_day[l]
            data_thisday = np.array(data_byday[day])
            data_x1s[k, 24 * l:24 * (l + 1), :] = data_thisday

        cn_predict_dict = getCnPredict()
        data_x2[k] = np.array(cn_predict_dict[date])

    data_x1s = torch.tensor(data_x1s)
    data_x2 = torch.tensor(data_x2)
    label_gt = torch.tensor(label_gt)

    data_x1s = data_x1s.float()
    data_x2 = data_x2.float()
    label_gt = label_gt.float()

    if torch.cuda.is_available():
        data_x1s = data_x1s.cuda()
        data_x2 = data_x2.cuda()
        label_gt = label_gt.cuda()
        lstm = lstm.cuda()

    output = lstm(data_x1s, data_x2, seq_num)


    result_list = output2result(output)

    print(output)

    print(result_list)


    result_agg = aggregateResultList(result_list)
