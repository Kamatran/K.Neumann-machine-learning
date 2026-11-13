"""A simple convolutional network (W6, task Z3)."""
from torch import nn


class SimpleCNN(nn.Module):
    """Blocks Conv2d(3x3, padding 1) -> ReLU -> MaxPool2d(2), one per entry of channels,
    then the classifier: Flatten -> Linear(hidden) -> ReLU -> Linear(n_classes).
    Every block keeps the height and width (padding 1) and the pooling halves them, so after three blocks
    a 64x64 image becomes 8x8 maps: the first Linear gets channels[-1] * 8 * 8 numbers.
    """
    def __init__(self, n_classes, channels=(16, 32, 64), hidden=64, size=64, in_channels=3):
        super().__init__()
        
        features = []
        current_channels = in_channels
        for out_channels in channels:
            features.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            current_channels = out_channels
        self.features = nn.Sequential(*features)
        
        num_pools = len(channels)
        final_size = size // (2 ** num_pools)
        features_out = channels[-1] * final_size * final_size
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features_out, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x