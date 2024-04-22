import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, activation=nn.ReLU()
    ):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = activation  # Store the activation function

    def forward(self, x):
        # Assuming x has shape [batch_size, seq_len, input_size]
        out, _ = self.rnn(x)
        
        # Handling output dimensions depending on whether it's 2D or 3D
        if out.dim() == 3:  # [batch_size, seq_len, hidden_size]
            out = out[:, -1, :]  # Take the last time step
        elif out.dim() != 2:  # [batch_size, hidden_size]
            raise ValueError("Unexpected output shape from RNN layer")

        out = self.fc(out)  # Apply the linear transformation
        out = self.activation(out)  # Apply the activation function
        return out
