from LSTM_20200524 import LSTM
import torch

import numpy as np

import pickle as pkl

import generate_gt_predict

from math import sqrt, acos, pi

import sys

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


if len(sys.argv) >= 2:
    DATE = str(sys.argv[1])
else:
    DATE = '20200617'

def getTabPredict(date):
    tab_predict_dict = {}
    with open('./crawled/weathertab_{}'.format(date), 'r') as f:
        head = f.readline()
        while True:
            line = f.readline()
            if line:
                date, rain_prob, high_temp_lc, high_temp_hc, low_temp_lc, low_temp_hc = line.split('\t')
                rain_prob = int(rain_prob) / 100.
                high_temp = (int(high_temp_lc) + int(high_temp_hc)) / 2.
                low_temp = (int(low_temp_lc) + int(low_temp_hc)) / 2.

                high_temp_norm = generate_gt_predict.normalize(high_temp, generate_gt_predict.TEMP_MAX_LC, generate_gt_predict.TEMP_MAX_HC)
                low_temp_norm = generate_gt_predict.normalize(low_temp, generate_gt_predict.TEMP_MIN_LC, generate_gt_predict.TEMP_MIN_HC)

                tab_predict_dict[date] = [low_temp_norm, high_temp_norm, rain_prob]

            else:
                break
    return tab_predict_dict


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

    return result_agg

    #
    # for i in range(len(result_agg) // 5):
    #     print('{}h, result = {}'.format(i, result_agg[i*5:(i+1)*5]))




if __name__ == '__main__':

    path_gt = 'outputs/gt_dict_20200514.pkl'
    gt_dict = load_object(path_gt)

    path_tab = './outputs/tab_predict_dict_20200528.pkl'
    tab_predict_dict_old = load_object(path_tab)

    path_model = './models/tab_lstm_epoch14_20200524.pkl'

    training_days = load_object('./outputs/training_days.pkl')
    data_byday = load_object('./outputs/data_normalized_byday_20200514.pkl')


    print(data_byday['2010-08-01'])


    seq_num = 31*24
    feature_num = 10
    flat_feature_num = 3
    output_num = 6*24

    BOOM_FACTOR = 1.618

    lstm = LSTM(hidden_size=64, feature_num=feature_num, flat_feature_num=flat_feature_num, output_num=output_num)
    lstm.load_state_dict(torch.load(path_model, map_location=lambda storage, loc: storage))

    print(lstm)

    tab_predict_dict = getTabPredict(DATE)

    output_results = {}

    for output_day in tab_predict_dict:

    # output_day = '2020-08-08'
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


            data_x2[k] = np.array(tab_predict_dict[date])

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

        knn_days = []
        for k in range(N):
            date, idx = training_samples[k]
            training_day = training_days[date[5:]][idx]
            # print(training_day)
            for l in range(len(training_day)):
                day = training_day[l]
                knn_days.append(day)

        print(knn_days)

        knn_distances_list = []
        print(tab_predict_dict[output_day])
        for date in knn_days:
            distance = np.sum(np.abs(np.array(tab_predict_dict_old[date]) - np.array(tab_predict_dict[output_day])))
            knn_distances_list.append(distance)

        knn_distances_list = np.array(knn_distances_list)
        arg_sort = np.argsort(knn_distances_list)
        print(knn_distances_list[arg_sort])

        topK = 5
        W = 20

        gt_num = 24 * 5

        topK_distance = knn_distances_list[arg_sort[:topK]]
        weights = np.exp(-W * topK_distance)
        weights /= np.sum(weights)
        print('the weight = {}'.format(weights))

        eta, etani = 0.9, 0.1
        final_results = np.zeros([gt_num], dtype=np.float64)
        for i in range(topK):
            final_results = final_results + weights[i] * np.array(gt_dict[knn_days[arg_sort[i]]])

        for j in range(gt_num):
            if j % 5 == 0:
                final_results[j] = reverseNormalize(final_results[j], generate_gt_predict.TEMP_MIN_LC, generate_gt_predict.TEMP_MIN_HC)
            elif j % 5 == 1:
                final_results[j] = reverseNormalize(final_results[j], generate_gt_predict.TEMP_MAX_LC, generate_gt_predict.TEMP_MAX_HC)
            elif j % 5 == 2:
                final_results[j] = final_results[j]
            elif j % 5 == 3:
                final_results[j] = BOOM_FACTOR * reverseNormalize(final_results[j], generate_gt_predict.WIND_SPEED_X_LC, generate_gt_predict.WIND_SPEED_X_HC)
            elif j % 5 == 4:
                final_results[j] = BOOM_FACTOR * reverseNormalize(final_results[j], generate_gt_predict.WIND_SPEED_Y_LC, generate_gt_predict.WIND_SPEED_Y_HC)

        final_results = eta * final_results + etani * np.array(result_agg)

        print(final_results)

        output_result = []
        print('following is the forecast for the date {}:'.format(output_day))
        for i in range(gt_num // 5):
            slices = final_results[i*5: (i+1)*5]
            wind_speed = sqrt(slices[3] * slices[3] + slices[4] * slices[4])
            direction = acos(slices[3] / wind_speed) * 180 / pi
            if slices[3] < 0:
                direction += 180

            print('in {}:00(UTC+0), temp_min = {}, temp_max = {}, rain_prob = {}, wind_speed = {}, wind_direction = {}'.format(i,
                            slices[0], slices[1], slices[2], wind_speed, direction))

            output_format = {'time': i, 'temp_min': slices[0], 'temp_max': slices[1], 'rain_prob': slices[2],
                             'wind_speed': wind_speed, 'wind_direction': direction}

            output_result.append(output_format)


        output_results[output_day] = output_result


    save_object(output_results, './outputs/output_results_20200612.pkl')

