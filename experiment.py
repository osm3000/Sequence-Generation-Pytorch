import sys
import matplotlib
# matplotlib.use('Agg')
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import data_generation
import models

batch_size = 64
seq_len = 100  # This is equivalent to time steps of the sequence in keras
input_size = 1
hidden_size = 51
target_size = 1
nb_samples = 1000
nb_epochs_mainTraining = 150
nb_epochs_fineTuning = 200

X_train, X_val, X_test, y_train, y_val, y_test = data_generation.generate_data(data_fn=data_generation.sine_1,
                                                                               nb_samples=nb_samples, seq_len=seq_len)
# exit()
rnn = models.lstm_rnn_gru(input_size=input_size, hidden_size=hidden_size).cuda()

loss_fn = nn.MSELoss()
optimizer = optim.RMSprop(rnn.parameters(), lr=0.00001, momentum=0.9)
# optimizer = optim.SGD(rnn.parameters(), lr=0.000003, momentum=0.95)
# optimizer = optim.LBFGS(rnn.parameters())
# optimizer = optim.Adam(rnn.parameters(), lr=0.00001)
# optimizer = optim.Adagrad(rnn.parameters(), lr=0.0001)
"""
Training with ground truth -- The input is the ground truth
"""
for epoch in range(nb_epochs_mainTraining):
    training_loss = 0
    val_loss = 0
    rnn.train(True)
    for batch, i in enumerate(range(0, X_train.size(0) - 1, batch_size)):
        data, targets = data_generation.get_batch(X_train, y_train, i, batch_size=batch_size)
        output = rnn(data)
        optimizer.zero_grad()
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        training_loss += loss.data[0]
    training_loss /= batch
    rnn.train(False)
    for batch, i in enumerate(range(0, X_val.size(0) - 1, batch_size)):
        data, targets = data_generation.get_batch(X_val, y_val, i, batch_size=batch_size)
        output = rnn(data)
        loss = loss_fn(output, targets)
        val_loss += loss.data[0]
    val_loss /= batch

    print "Ground truth - Epoch " + str(epoch) + " -- train loss = " + str(training_loss) + " -- val loss = " + str(val_loss)

"""
Gerard idea: Fine tuning with prediction
In this case, the input will be the prediction from the previous point
The idea is to make the model more robust
"""
# optimizer = optim.RMSprop(rnn.parameters(), lr=0.00001, momentum=0.9) # In order to reset the optimizer
# for epoch in range(nb_epochs_fineTuning):
#     training_loss = 0
#     val_loss = 0
#     rnn.train(True)
#     # data, _ = data_generation.get_batch(X_train, y_train, 0, batch_size=batch_size)
#     for batch, i in enumerate(range(0, X_train.size(0) - 1, batch_size)):
#         # print i
#         data, targets = data_generation.get_batch(X_train, y_train, i, batch_size=batch_size)
#         # output = rnn(data)
#         output = rnn.fineTuneModel(data, start_steps=5)
#         if targets.size() != output.size():
#             """
#             Dirty solution to avoid the hazardous last batch
#             """
#             break
#         optimizer.zero_grad()
#         loss = loss_fn(output, targets)
#         loss.backward()
#         optimizer.step()
#         training_loss += loss.data[0]
#         # Drop the first element of the sequence, and concatenate the last element
#         # data = data[:, 1:]
#         # data = torch.cat((data, output[:, -1:]), 1)
#         # data = Variable(data.data) # To repackage the data, and remove the need for its history
#
#     training_loss /= batch
#     rnn.train(False)
#     for batch, i in enumerate(range(0, X_val.size(0) - 1, batch_size)):
#         data, targets = data_generation.get_batch(X_val, y_val, i, batch_size=batch_size)
#         output = rnn(data)
#         loss = loss_fn(output, targets)
#         val_loss += loss.data[0]
#     val_loss /= batch
#
#     print "Model tuning - Epoch " + str(epoch) + " -- train loss = " + str(training_loss) + " -- val loss = " + str(val_loss)

"""
Measuring the test score -> running the test data on the model
"""
rnn.train(False)
test_loss = 0
list1 = []
list2 = []
for batch, i in enumerate(range(0, X_test.size(0) - 1, batch_size)):
    data, targets = data_generation.get_batch(X_test, y_test, i, batch_size=batch_size)
    output = rnn(data)
    loss = loss_fn(output, targets)
    test_loss += loss.data[0]
    target_last_point = torch.squeeze(targets[:, -1]).data.cpu().numpy().tolist()
    pred_last_point = torch.squeeze(output[:, -1]).data.cpu().numpy().tolist()
    list1 += target_last_point
    list2 += pred_last_point
plt.figure()
plt.plot(list1, "b")
plt.plot(list2, "r")
plt.legend(["Original data", "Generated data"])
plt.show()
test_loss /= batch
print "Test loss = ", test_loss

"""
Generating sequences - attempt 1 --> Exactly like "time sequence prediction" example
"""
# data = X_test[0, :].view(1, -1)
# output = rnn(data, future=300)
# output = torch.squeeze(output).data.cpu().numpy()
# plt.figure()
# plt.plot(output)
# plt.xlabel("Time step")
# plt.ylabel("Signal amplitude")
# plt.show()
"""
Generating sequences - attempt 2 --> Concatenating the output with the input, and feed the new data point to the model.
I get a point from the test data as a starting point (a seed), and I ask the model to continue generation after this seed
"""
data = X_test[0, :].view(1, -1)
final = []
for i in range(300):
    # print data.view(1, -1).data.cpu().numpy().tolist()
    output = rnn(data)
    outputx = torch.squeeze(output).data.cpu().numpy()
    datax = torch.squeeze(data).data.cpu().numpy()
    final.append(outputx[-1])
    data = data[:, 1:]
    data = torch.cat((data, output[:, -1].view(1, 1)), 1)
print "---------------------------------------"
print final
plt.figure()
plt.plot(final)
plt.show()
