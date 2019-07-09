import torch.nn as nn
import torch


class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class=10):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                            batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x1, x2):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # print(x)
        out1, _ = self.lstm(x1)
        out2, _ = self.lstm(x2)
        print(out1.size())
        print(out2.size())


model = Rnn(28, 128, 2)
x1 = torch.rand((3, 24, 28))
x2 = torch.rand((3, 12, 28))
model(x1, x2)
