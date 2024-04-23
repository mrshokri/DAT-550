import torch.nn as nn


class FFNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, activation=nn.ReLU()
    ):
        super(FFNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            elif i == num_layers - 1:
                self.hidden_layers.append(nn.Linear(hidden_size, output_size))
            else:
                self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(activation)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x
