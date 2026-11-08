"""Multilayer perceptron for images (W5, task Z1).

For now an image is flattened: nn.Flatten turns (3, 64, 64) into 12 288 separate numbers. The network does
not know which pixels are neighbours, a pattern learned in one place does not work in another, and the first
layer has millions of weights. From W6 convolutional layers come before the MLP and remove these limits;
the MLP stays as the last part of the network.

Check it with:
    uv run pytest exercises/w05
"""

from torch import nn


class MLP(nn.Module):
    """Flatten -> [Linear -> activation] for every hidden width -> Linear to n_classes.

    in_features: number of input values of one example (e.g. 3 * 64 * 64);
    hidden: widths of the hidden layers, e.g. (256,) or (512, 256);
    activation: a class such as nn.ReLU, or None for a stack of linear layers without non-linearity.
    Keep the layers in an nn.Sequential stored as self.net.
    """

    def __init__(self, in_features, n_classes, hidden=(256,), activation=nn.ReLU):
        super().__init__()
        
        layers = [nn.Flatten()]
        current_features = in_features
        
        for width in hidden:
            layers.append(nn.Linear(current_features, width))
            
            if activation is not None:
                layers.append(activation())
    
            current_features = width
            
        layers.append(nn.Linear(current_features, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())