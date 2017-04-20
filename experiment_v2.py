import matplotlib
# matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import data_generation
import models
import itertools
"""
Items I want to loop over:
- seq_len
- signal_frequency [30, 60, 90, 180]
- hidden_size [20, 50, 100, 200]
- functions [sine1, sine2, sine3]
- future steps = 1000
- with and without noise
"""
batch_size = 64
seq_len = 500  # This is equivalent to time steps of the sequence in keras
input_size = 1
hidden_size = 51
target_size = 1
nb_samples = 1000
nb_epochs_mainTraining = 150
nb_epochs_fineTuning = 200
future_steps = 1000
gen_seed = 100
config = {"max_epochs": [200],
          "future_steps": [1000],
          # "functions": [data_generation.sine_1, data_generation.sine_2, data_generation.sine_3],
          "signal_freq": [30., 60., 90., 180.],
          "seq_len": [100, 200],
          "add_noise": [True, False],
          "ES_criteria": [0.015],
          "hidden_size": [20, 50, 100, 200]}

# result_list = map(dict, itertools.combinations(config.iteritems(), 1))
# result_list = map(dict, itertools.product(config.iteritems()))
# print result_list
print config.iteritems().next()
exit()
def run_experiment(configurations):
    X_train, X_val, X_test, y_train, y_val, y_test = data_generation.generate_data(data_fn=data_generation.sine_2,
                                                                                   nb_samples=nb_samples, seq_len=seq_len,
                                                                                   signal_freq=60., add_noise=False)
    rnn = models.lstm_rnn_gru(input_size=input_size, hidden_size=hidden_size, cell_type="lstm").cuda()
    for module in rnn.modules():
        print module
    loss_fn = nn.MSELoss()
    optimizer = optim.RMSprop(rnn.parameters(), lr=0.00001, momentum=0.9)
    """
    Training with ground truth -- The input is the ground truth
    """
    try:
        val_loss_list = []
        for epoch in range(nb_epochs_mainTraining):
            training_loss = 0
            val_loss = 0
            rnn.train(True)
            for batch, i in enumerate(range(0, X_train.size(0) - 1, batch_size)):
                data, targets = data_generation.get_batch(X_train, y_train, i, batch_size=batch_size)
                output = rnn(data) # This is the original, training with ground truth only
                # output = rnn(data[:, :100], future=400)  # Here, trying to use the model output as part of the input
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
            val_loss_list.append(val_loss)
            print "Ground truth - Epoch " + str(epoch) + " -- train loss = " + str(training_loss) + " -- val loss = " + str(val_loss)

            # Early stopping condition -- when the last 4 epochs results in a validation error < 0.015
            cond_true = False
            if len(val_loss_list) >= 4:
                cond_true = True
                for i in val_loss_list[-4:]:
                    if i > 0.015:
                        cond_true = False
                        break
            if cond_true == True:
                print "Triggering early stopping criteria - Out of training"
                break
    except KeyboardInterrupt:
        print "Early stopping for the training"

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
    data = X_test[0, :].view(1, -1)
    # output = rnn(data[:, :100], future=future_steps)
    output = rnn(data, future=future_steps)
    output = torch.squeeze(output).data.cpu().numpy()
    plt.figure()
    plt.plot(output)
    plt.xlabel("Time step")
    plt.ylabel("Signal amplitude")
    plt.show()
