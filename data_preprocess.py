from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

import numpy as np

import torch

def load_data():
    ################################
    # DataLoader
    ################################

    # set up fields
    TEXT = data.Field()
    LABEL = data.Field(sequential=False, dtype=torch.long)

    # make splits for data
    # DO NOT MODIFY: fine_grained=True, train_subtrees=False
    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=True, train_subtrees=False)

    # print information about the data
    print('train.fields', train.fields)
    print('len(train)', len(train))
    print(len(val))
    print(len(test))
    print('vars(train[0])', vars(train[1]))

    # build the vocabulary
    # you can use other pretrained vectors, refer to https://github.com/pytorch/text/blob/master/torchtext/vocab.py
    TEXT.build_vocab(train, vectors=Vectors(name='vector.txt', cache='./data'))
    LABEL.build_vocab(train)
    # We can also see the vocabulary directly using either the stoi (string to int) or itos (int to string) method.
    print(TEXT.vocab.itos[:10])
    print(LABEL.vocab.stoi)
    print(TEXT.vocab.freqs.most_common(20))

    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())
    # print(TEXT.vocab.vectors[8, :])

    # make iterator for splits
    return train, val, test, TEXT.vocab.vectors

# my generator to get the next epoch
def next_batch(train_data, train_labels, train_lengths, batch_size):
    size = len(train_labels)
    slices = size // batch_size #in one epoch the iter number
    while True:
        indexArray = np.array(range(size), dtype=int)  # 下标数组
        np.random.shuffle(indexArray) # shuffle一下index
        for i in range(slices):
            low = i * batch_size
            high = (i+1) * batch_size
            x, y, lengths = train_data[indexArray[low:high],:, :], train_labels[indexArray[low:high]], train_lengths[indexArray[low:high]]
            inx = torch.from_numpy(np.argsort(lengths.numpy())[::-1].copy())
            yield  x[inx, :, :], y[inx], lengths[inx] # sort in decrease order

def organize_vector(data, pretrained_embeddings, word2vec_size):
    seq_len = data.shape[0]
    size = data.shape[1]
    output_vector = torch.zeros(size, seq_len, word2vec_size)

    for i in range(size):
        for t in range(seq_len):
            output_vector[i][t][:] = pretrained_embeddings[data[t][i], :]

    return output_vector

def get_lengths(data):
    data_numpy = data.numpy()
    lengths = [np.append(np.where(data_numpy[:, i] == 1)[0], data.shape[0] - 1)[0] + 1 for i in
                     range(data.shape[1])]
    lengths = torch.from_numpy(np.array(lengths))
    print(lengths)
    return lengths

if __name__ == '__main__':
    batch_size = 64

    train, val, test, pretrained_embeddings = load_data()
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size)

    train_iter_total, val_iter_total, test_iter_total = data.BucketIterator.splits((train, val, test), batch_size=len(train))
    train_data = next(iter(train_iter_total))
    val_data = next(iter(val_iter_total))
    test_data = next(iter(test_iter_total))
    print(train_data.text)
    print(train_data.text.shape)
    print(val_data.text.shape)
    print(test_data.text.shape)

    train_lengths = get_lengths(train_data.text)

    # print batch information
    batch = next(iter(train_iter))  # for batch in train_iter
    print(batch.text)  # input sequence
    print(batch.label)  # groud truth
    print(batch.text.shape)
    print(batch.label.shape)



    print(pretrained_embeddings.shape)