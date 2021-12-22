import torch
import numpy as np

import matplotlib.pyplot as plt

def train(net, train_batch_genitor, criterion, optimizer, train_data_x, train_data_labels,
          train_lengths, val_data_x, val_data_labels, val_lengths, epoch_max, iter_per_epoch, batch_size):
    loss_list = np.zeros(epoch_max)
    accuracy_list = np.zeros([2, epoch_max])

    for epoch in range(epoch_max):
        for iter_num in range(iter_per_epoch):
            batch_input, label_gt, batch_lengths = train_batch_genitor.__next__()

            # print('batch_input.shape = {}'.format(batch_input.shape))

            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                label_gt = label_gt.cuda()

            output = net(batch_input, batch_lengths)
            loss = criterion(output, label_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter_num + 1) % 10 == 0:
                accuracy = float(sum(label_gt == output.argmax(dim=1))) / batch_size
                print('epoch {} iter {}, train accuaracy = {}'.format(epoch, iter_num+1, accuracy))

        output_total = net(train_data_x, train_lengths)
        loss_list[epoch] = criterion(output_total, train_data_labels)
        accuracy_list[0, epoch] = sum(train_data_labels.cpu().numpy() == output_total.argmax(dim=1).cpu().numpy()) / float(len(train_data_labels))

        accuracy_list[1, epoch] = sum(val_data_labels.cpu().numpy() == net(val_data_x, val_lengths).argmax(dim=1).cpu().numpy()) / float(len(val_data_labels))
        print('epoch {}, loss = {}, train accuracy = {}, val accuracy = {}'.format(epoch, loss_list[epoch], accuracy_list[0, epoch], accuracy_list[1, epoch]))


    return accuracy_list, loss_list



def test(net, test_data_x, test_data_labels, test_lengths):
    test_accuracy = sum(test_data_labels.cpu().numpy() == net(test_data_x, test_lengths).argmax(dim=1).cpu().numpy()) \
                    / float(len(test_data_labels))
    print('the test accuracy = {}'.format(test_accuracy))
    return test_accuracy


def multi_run(Net, train_batch_genitor, criterion, train_data_x, train_data_labels,
          train_lengths, val_data_x, val_data_labels, val_lengths, test_data_x, test_data_labels, test_lengths,
          epoch_max, iter_per_epoch, batch_size, learning_rate, times,
          written=False, written_file='./run.csv', hidden_size=256, drop_out=0):
    final_accuracy_list = np.zeros([3, times])

    for k in range(times):
        net = Net(hidden_size=hidden_size, dropout=drop_out)

        if torch.cuda.is_available():
            net = net.cuda()

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        accuracy_list, _ = train(net, train_batch_genitor, criterion, optimizer, train_data_x, train_data_labels,
                                 train_lengths, val_data_x, val_data_labels, val_lengths, epoch_max, iter_per_epoch,
                                 batch_size)

        final_accuracy_list[0, k] = accuracy_list[0,epoch_max-1]
        final_accuracy_list[1, k] = accuracy_list[1,epoch_max-1]

        final_accuracy_list[2, k] = test(net, test_data_x, test_data_labels, test_lengths)

    if written:
        file = open(written_file, 'w')
        types = ['Train', 'Validate', 'Test']

        line_1 = ','+ ','.join([str(i) for i in range(1,times+1)]) + ',avg'
        print(line_1)
        file.write(line_1+'\n')

        for k in range(3):
            line_k = types[k] + ',' + ','.join([str(acc) for acc in final_accuracy_list[k, :]]) + ',' + str(np.mean(final_accuracy_list[k,:]))
            print(line_k)
            file.write(line_k+'\n')

        file.close()

    return final_accuracy_list