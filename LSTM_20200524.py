import sys
sys.path.append("/work/czm/deep_learning/machine_learning")


from torch import nn
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl

import os

from generate_pm15_days import nextday

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean()



class LSTM(nn.Module):
    def __init__(self, hidden_size=10, dropout=0, feature_num=3, flat_feature_num=4, output_num=6):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.flat_feature_num = flat_feature_num

        self.lstm = nn.LSTM(
            input_size=feature_num,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout
        ) # shape - (batch, seq, feature)

        self.out = nn.Linear(hidden_size+flat_feature_num, output_num)

    def forward(self, x1s, x2, length): # x1s: a sequence shape as (batch, seq, feature), x2: others feature shape as (batch, flag_feature)
        # x = pack_padded_sequence(x, lengths, batch_first=True)

        r_out, (h_n, h_c) = self.lstm(x1s, None)
        # r_out, _ = pad_packed_sequence(r_out, batch_first=True)
        r_out_cpu = r_out.cpu()

        # print(r_out_cpu)

        batch_size = r_out_cpu.shape[0]

        r_selected = torch.zeros(batch_size, self.hidden_size+self.flat_feature_num)
        for i in range(batch_size):
            r_selected[i, :self.hidden_size] = r_out_cpu[i, length-1, :]
            r_selected[i, self.hidden_size:] = x2[i, :]

        if torch.cuda.is_available():
            r_selected = r_selected.cuda()

        out = self.out(r_selected)

        out = out.cpu()

        # out = torch.exp(out) / (torch.exp(out) + 1)
        return out


if __name__ == '__main__':
    data_byday = load_object('./outputs/data_normalized_byday_20200514.pkl')

    training_days = load_object('./outputs/training_days.pkl')
    tab_predict_dict = load_object('./outputs/tab_predict_dict_20200524.pkl')
    gt_dict = load_object('./outputs/gt_dict_20200524.pkl')

    start_day = '1979-03-01'
    end_day = '2020-03-14'

    training_samples = []
    while start_day <= end_day:
        date_noyear = start_day[5:]
        # print(date_noyear)

        train_thisday = training_days[date_noyear]

        for i in range(len(train_thisday)):
            if train_thisday[i][-1] < start_day:
                # print(train_thisday[i][-1], start_day)
                training_samples.append([start_day, i])

        start_day = nextday(start_day)

    print(len(training_samples))



    # print(training_days['01-01'])
    #
    #
    print(data_byday['2020-02-29'])
    print(len(data_byday['2020-02-29']), len(data_byday['2020-02-20'][0]))



    feature_num = len(data_byday['2020-02-29'][0])
    flat_feature_num = len(tab_predict_dict['2020-02-29'])
    output_num = 6 * 24
    learning_rate = 0.01

    print(feature_num)
    print(flat_feature_num)

    lstm = LSTM(hidden_size=64, feature_num=feature_num, flat_feature_num=flat_feature_num, output_num=output_num)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    batch_size = 50
    seq_num = 31 * 24

    N = len(training_samples)

    lstm = lstm.float()

    EPOCH_NUM = 20
    for i in range(EPOCH_NUM):
        indexes = np.array(range(N))
        np.random.shuffle(indexes)
        for j in range(N // batch_size):
            data_x1s = np.zeros([batch_size, seq_num, feature_num], dtype=np.double)
            data_x2 = np.zeros([batch_size, flat_feature_num], dtype=np.double)
            label_gt = np.zeros([batch_size, output_num], dtype=np.double)
            for k in range(batch_size):
                sample_idx = indexes[j*batch_size+k]
                date, idx = training_samples[sample_idx]
                training_day = training_days[date[5:]][idx]
                # print(training_day)
                for l in range(len(training_day)):
                    day = training_day[l]
                    data_thisday = np.array(data_byday[day])
                    data_x1s[k, 24*l:24*(l+1), :] = data_thisday

                data_x2[k] = np.array(tab_predict_dict[date])
                label_gt[k] = np.array(gt_dict[date])

            data_x1s = torch.from_numpy(data_x1s)
            data_x2 = torch.from_numpy(data_x2)
            label_gt = torch.from_numpy(label_gt)

            data_x1s = data_x1s.float()
            data_x2 = data_x2.float()
            label_gt = label_gt.float()

            if torch.cuda.is_available():
                data_x1s = data_x1s.cuda()
                data_x2 = data_x2.cuda()
                label_gt = label_gt.cuda()
                lstm = lstm.cuda()

            output = lstm(data_x1s, data_x2, seq_num)
            loss = criterion(output, label_gt.cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch #{}, slice #{}, loss = {}'.format(i, j, loss))
            # print(output)
            # print()


        if not os.path.exists('./models'):
            os.makedirs('./models')

        torch.save(lstm.state_dict(), './models/tab_lstm_epoch{}_20200524.pkl'.format(i))



    # label_gt = torch.tensor([[1.0, 0., -1., 1., 0.]])
    #
    #
    # print(lstm)
    #
    # data_x1s = torch.tensor([[[-0.4, 0.4, 0.5], [0.2, -0.1, 0.4]]])
    # data_x2 = torch.tensor([[0.3, -0.5, 0.2, -0.4]])
    # print(lstm(data_x1s, data_x2, 1))
    #
    # for i in range(10):
    #     if torch.cuda.is_available():
    #         label_gt = label_gt.cuda()
    #
    #     output = lstm(data_x1s, data_x2, 1)
    #     loss = criterion(output, label_gt)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()



