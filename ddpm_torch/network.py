import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F

def mlp(input_dim, hidden_dim, hidden_depth, output_dim, initialize):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    
    for _ in range(hidden_depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    
    output_layer = nn.Linear(hidden_dim, output_dim)
    
    if initialize:
        _initialize_weights(output_layer)

    layers.append(output_layer)

    return nn.Sequential(*layers)

def _initialize_weights(output_layer):
    # Manually set weights for the last layer to get the desired initialization
    nn.init.constant_(output_layer.weight, 0.0)
    nn.init.constant_(output_layer.bias, 0.5)

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_depth, output_dim=1, initialize=False):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.output_dim = output_dim

        self.mlps = mlp(input_dim, hidden_dim, hidden_depth, output_dim, initialize)

    def forward(self, input):
        out = self.mlps(input)

        return out

class ValueNetwork(nn.Module):
    def __init__(self, input_channels, hidden_dim, hidden_depth, output_dim=1, initialize=False):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_depth = hidden_depth

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        conv_output_size = 128 * 4 * 4  # Assuming input size 32x32

        self.mlps = mlp(conv_output_size, hidden_dim, hidden_depth, output_dim, initialize)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(x.size(0), -1)  # Flatten the output of the conv layers
        out = self.mlps(x)
        
        return out

class ActorNetwork(nn.Module):
    def __init__(self, input_channels, hidden_dim, hidden_depth, output_dim=2, initialize=True):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_depth = hidden_depth

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        conv_output_size = 128 * 4 * 4  # Assuming input size 32x32

        self.mlps = mlp(conv_output_size, hidden_dim, hidden_depth, output_dim, initialize)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(x.size(0), -1)  # Flatten the output of the conv layers
        out = self.mlps(x)
        
        alpha, beta = torch.chunk(out, 2, dim=-1)  # Split into alpha and beta
        alpha = F.softplus(alpha) + 1e-5  # Ensure alpha > 0
        beta = F.softplus(beta) + 1e-5  # Ensure beta > 0
        
        return alpha, beta
