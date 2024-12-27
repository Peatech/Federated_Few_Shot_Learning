# few_shot/models.py

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict

##########
# Layers #
##########
class Flatten(nn.Module):
    """Flatten [B, d1, d2, ...] -> [B, d1*d2*...]"""
    def forward(self, input):
        return input.view(input.size(0), -1)


class GlobalMaxPool1d(nn.Module):
    def forward(self, input):
        return F.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool2d(nn.Module):
    def forward(self, input):
        return F.avg_pool2d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """3x3 Conv + BN + ReLU + 2x2 MaxPool."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def functional_conv_block(x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor,
                          bn_weights, bn_biases) -> torch.Tensor:
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None,
                     weight=bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x

##########
# Models #
##########
def get_few_shot_encoder(num_input_channels=1) -> nn.Module:
    """Creates a 4-layer CNN + Flatten, used by Matching/Prototypical Nets."""
    return nn.Sequential(
        conv_block(num_input_channels, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        Flatten(),
    )

class FewShotClassifier(nn.Module):
    """
    Example classifier used in MAML or prototypical approaches.
    Not directly used by Matching Networks, but might remain for reference.
    """
    def __init__(self, num_input_channels: int, k_way: int, final_layer_size: int = 64):
        super().__init__()
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.logits = nn.Linear(final_layer_size, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.logits(x)

    def functional_forward(self, x, weights):
        for block in [1, 2, 3, 4]:
            x = functional_conv_block(
                x,
                weights[f'conv{block}.0.weight'],
                weights[f'conv{block}.0.bias'],
                weights.get(f'conv{block}.1.weight'),
                weights.get(f'conv{block}.1.bias')
            )
        x = x.view(x.size(0), -1)
        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])
        return x

class MatchingNetwork(nn.Module):
    def __init__(self, n: int, k: int, q: int, fce: bool,
                 num_input_channels: int,
                 lstm_layers: int, lstm_input_size: int,
                 unrolling_steps: int, device: torch.device):
        """
        Matching Networks (Vinyals et al., 2016).
        """
        super().__init__()
        self.n = n
        self.k = k
        self.q = q
        self.fce = fce
        self.num_input_channels = num_input_channels

        # Encoder is a standard 4-layer CNN
        self.encoder = get_few_shot_encoder(self.num_input_channels)

        # If FCE is True, add the LSTMs
        if self.fce:
            self.g = BidirectionalLSTM(lstm_input_size, lstm_layers).to(device, dtype=torch.double)
            self.f = AttentionLSTM(lstm_input_size, unrolling_steps=unrolling_steps).to(device, dtype=torch.double)

    def forward(self, inputs):
        """
        Typically unused because the training logic is in matching_net_episode. 
        But we might define a pass if we want a direct forward approach.
        """
        # Minimal usage: 
        return self.encoder(inputs)


class BidirectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """
        Bidirectional LSTM for generating fully conditional embeddings (g).
        Size = input_size = hidden_size for skip connections.
        """
        super().__init__()
        self.num_layers = layers
        self.batch_size = 1
        self.lstm = nn.LSTM(
            input_size=size,
            hidden_size=size,
            num_layers=layers,
            bidirectional=True
        )

    def forward(self, inputs):
        """
        inputs: shape (seq_len, batch, input_size)
        We'll do skip connection: output += inputs
        """
        output, (hn, cn) = self.lstm(inputs, None)

        # Separate forward & backward outputs
        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # skip connection
        combined = forward_output + backward_output + inputs
        return combined, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int):
        """
        Attentional LSTM for query set embeddings (f).
        """
        super().__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size, hidden_size=size)

    def forward(self, support, queries):
        """
        support: shape (k*n, embed_dim)
        queries: shape (q*k, embed_dim)
        """
        if support.shape[-1] != queries.shape[-1]:
            raise ValueError("Support/query embedding dims differ!")

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        device = queries.device
        h_hat = torch.zeros_like(queries, device=device, dtype=queries.dtype)
        c = torch.zeros(batch_size, embedding_dim, device=device, dtype=queries.dtype)

        for _ in range(self.unrolling_steps):
            h = h_hat + queries

            # attention between h and support
            attentions = torch.mm(h, support.t()).softmax(dim=1)
            readout = torch.mm(attentions, support)

            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries
        return h
