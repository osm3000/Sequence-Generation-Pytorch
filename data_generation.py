import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

def get_batch(X, Y, i, batch_size=32):
    min_len = min(batch_size, len(X) - 1 - i)
    data = X[i:i + min_len, :]
    target = Y[i:i + min_len, :]
    return data, target

def sine_1(X, add_noise=False, noise_range=(-0.1, 0.1)):
    if add_noise:
        clean_signal = np.sin(2 * np.pi * (X) / 60.)
        noisy_signal = clean_signal + np.random.uniform(noise_range[0], noise_range[1], size=clean_signal.shape)
        return noisy_signal
    else:
        # return np.sin(2 * np.pi * (X) / 180.)
        return np.sin(2 * np.pi * (X) / 60.)


def sine_2(X, add_noise=False, noise_range=(-0.1, 0.1)):
    if add_noise:
        clean_signal = (np.sin(2 * np.pi * (X) / 180.) + np.sin(2 * 2 * np.pi * (X) / 180.)) / 2.0
        noisy_signal = clean_signal + np.random.uniform(noise_range[0], noise_range[1], size=clean_signal.shape)
        return noisy_signal
    else:
        # return (np.sin(2 * np.pi * (X) / 180.) + np.sin(2 * 2 * np.pi * (X) / 180.)) / 2.0
        return (np.sin(2 * np.pi * (X) / 30.) + np.sin(2 * 2 * np.pi * (X) / 30.)) / 2.0


def sine_3(X, add_noise=False, noise_range=(-0.1, 0.1)):
    if add_noise:
        clean_signal = (np.sin(2 * np.pi * (X) / 180.) + np.sin(2 * 2 * np.pi * (X) / 180.) + np.sin(
        2 * 3 * np.pi * (X) / 180.)) / 3.0
        noisy_signal = clean_signal + np.random.uniform(noise_range[0], noise_range[1], size=clean_signal.shape)
        return noisy_signal
    else:
        # return (np.sin(2 * np.pi * (X) / 180.) + np.sin(2 * 2 * np.pi * (X) / 180.) + np.sin(
        # 2 * 3 * np.pi * (X) / 180.)) / 3.0
        return (np.sin(2 * np.pi * (X) / 90.) + np.sin(2 * 2 * np.pi * (X) / 90.) + np.sin(
            2 * 3 * np.pi * (X) / 90.)) / 3.0


def generate_data(data_fn=sine_1, nb_samples=10000, seq_len=100):
    x_data = np.arange(nb_samples)
    y_data = data_fn(x_data, add_noise=False)
    # y_data = data_fn(x_data, add_noise=True)
    plt.figure("Synthetic data", figsize=(15, 10))
    plt.title("Synthetic data")
    plt.plot(x_data[:400], y_data[:400])
    plt.savefig("synthetic_data.png")
    plt.close()

    X = []
    y = []
    # for i in range(0, y_data.shape[0] - 1, seq_len):
    for i in range(0, y_data.shape[0] - 1):
        if i + seq_len < y_data.shape[0]:
            X.append(y_data[i:i + seq_len])
            y.append(y_data[i + 1:i + seq_len + 1])  # next sequence (including the next point in the last)
            # y.append(y_data[i + seq_len])  # next point only
    X = np.array(X)
    y = np.array(y)

    # plt.figure(figsize=(15, 10))
    # n = np.squeeze(X[:, -1])
    # plt.plot(n, "go")
    # plt.show()

    nb_seq = X.shape[0]
    # 70% training, 10% validation, 20% testing
    train_sep = int(nb_seq * 0.7)
    val_sep = train_sep + int(nb_seq * 0.1)

    X_train = Variable(torch.from_numpy(X[:train_sep, :]).float()).cuda()
    y_train = Variable(torch.from_numpy(y[:train_sep, :]).float()).cuda()

    X_val = Variable(torch.from_numpy(X[train_sep:val_sep, :]).float()).cuda()
    y_val = Variable(torch.from_numpy(y[train_sep:val_sep, :]).float()).cuda()

    X_test = Variable(torch.from_numpy(X[val_sep:, :]).float()).cuda()
    y_test = Variable(torch.from_numpy(y[val_sep:, :]).float()).cuda()

    print ("X_train size = {}, X_val size = {}, X_test size = {}".format(X_train.size(), X_val.size(), X_test.size()))

    return X_train, X_val, X_test, y_train, y_val, y_test


# generate_data(data_fn=sine_1)
