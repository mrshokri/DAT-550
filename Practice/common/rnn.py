import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Assuming x has shape [batch_size, seq_len, input_size]
        out, _ = self.rnn(x)
        # If out has shape [batch_size, seq_len, hidden_size] and seq_len is treated as 1
        # We need to ensure the tensor is correctly shaped for the linear layer
        # If the sequence length is 1, out will effectively be 2D after the rnn layer
        if out.dim() == 3:  # [batch_size, seq_len, hidden_size]
            out = out[:, -1, :]  # Get the outputs of the last time step
        elif out.dim() == 2:  # [batch_size, hidden_size]
            # No need to index, as there's no sequence length dimension
            pass  # out is already correctly shaped
        else:
            raise ValueError("Unexpected output shape from RNN layer")

        out = self.fc(out)
        return out
