import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if gpu is available use gpu


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,  # (batch_size, seq_length, input_dim)
            hidden_size=hidden_dim,  # (batch_size, seq_length, hidden_dim)
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,  # (batch_size, seq_length, hidden_dim)
            hidden_size=hidden_dim,  # (batch_size, seq_length, input_dim)
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Linear layer to transform final step of decoder to original space
        self.linear = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, input_dim)

    def forward(self, x):  # x: (batch_size, seq_length, input_dim)
        # Encoder
        _, (hidden_state, cell_state) = self.encoder(x)  # hidden_state: (num_layers, batch_size, hidden_dim)

        # Use the encoder's hidden state as the initial hidden state for the decoder
        seq_len = x.size(1)
        context_vector = hidden_state.repeat(seq_len, 1, 1).transpose(0, 1)  # context_vector: (batch_size, seq_length, hidden_dim)

        # Decoder
        x_decoded, _ = self.decoder(context_vector,
                                    (hidden_state, cell_state))  # x_decoded: (batch_size, seq_length, input_dim)

        # If using bidirectional LSTM, transform to original space
        x_decoded = self.linear(x_decoded)  # x_decoded: (batch_size, seq_length, input_dim)

        return x_decoded

