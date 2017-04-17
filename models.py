import torch
import torch.nn as nn
from torch.autograd import Variable


class lstm_rnn_gru(nn.Module):
    def __init__(self, cell_type="lstm", input_size=1, hidden_size=20, output_size=1, nonlinearity="tanh"):
        super(lstm_rnn_gru, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nonlinearity = nonlinearity.lower()
        assert self.nonlinearity in ['tanh', 'relu']

        self.cell_type = cell_type.lower()
        if self.cell_type == "lstm":
            self.layer1 = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
            self.layer2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.output_size)
        elif self.cell_type == "rnn":
            self.layer1 = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size, nonlinearity=self.nonlinearity)
            self.layer2 = nn.RNNCell(input_size=self.hidden_size, hidden_size=self.output_size, nonlinearity=self.nonlinearity)
        elif self.cell_type == "gru":
            self.layer1 = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
            self.layer2 = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.output_size)
        else:
            raise ("Please enter a good cell type (LSTM/RNN/GRU)")

        self.layer1.weight_hh.data.normal_(0.0, 0.1)
        self.layer1.weight_ih.data.normal_(0.0, 0.1)
        self.layer2.weight_hh.data.normal_(0.0, 0.1)
        self.layer2.weight_ih.data.normal_(0.0, 0.1)

        # Should I do something about the biases here?

    def forward(self, input_data, future=0):
        """
        Note here that the input is 2-D
        :param input_data:
        :param future:
        :return:
        """
        outputs = []
        h_t = Variable(torch.zeros(input_data.size(0), self.hidden_size), requires_grad=False).cuda()
        h_t2 = Variable(torch.zeros(input_data.size(0), self.output_size), requires_grad=False).cuda()

        if self.cell_type == "lstm":
            c_t = Variable(torch.zeros(input_data.size(0), self.hidden_size), requires_grad=False).cuda()
            c_t2 = Variable(torch.zeros(input_data.size(0), self.output_size), requires_grad=False).cuda()

        # I've to feed the network one time step at a time
        for i, input_t in enumerate(input_data.chunk(input_data.size(1), dim=1)):
            if self.cell_type == "lstm":
                h_t, c_t = self.layer1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.layer2(c_t, (h_t2, c_t2))
                outputs += [c_t2]
            else:
                h_t = self.layer1(input_t, h_t)
                h_t2 = self.layer2(h_t, h_t2)
                outputs += [h_t2]

        for i in range(future):  # if we should predict the future
            if self.cell_type == "lstm":
                h_t, c_t = self.layer1(c_t2, (h_t, c_t))
                h_t2, c_t2 = self.layer2(c_t, (h_t2, c_t2))
                outputs += [c_t2]
            else:
                h_t = self.layer1(h_t2, h_t)
                h_t2 = self.layer2(h_t, h_t2)
                outputs += [h_t2]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
