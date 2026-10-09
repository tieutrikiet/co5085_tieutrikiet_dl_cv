import torch
import torch.nn as nn

from .config import IMAGE_SHAPE, INPUT_DIM, NUM_CLASSES


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.flatten = nn.Flatten()                   # [B, 1, H, W] -> [B, H*W]
        self.fc = nn.Linear(input_dim, num_classes)   # [10, 784], b: [10]

    def forward(self, x):
        return self.fc(self.flatten(x))               # logits z, no softmax yet

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.softmax(self(x), dim=1)          # P(y=i|x) = exp(z_i) / sum_j exp(z_j)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dims=(512, 256), num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        layers = [nn.Flatten()]                             # [B,1,28,28] -> [B,784]
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))  # z = xW + b
            layers.append(nn.ReLU())                        # a = ReLU(z)
            layers.append(nn.Dropout(dropout))              # dropout against overfitting
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))     # logits
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)                                # no softmax: CrossEntropyLoss applies it

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.softmax(self(x), dim=1)


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_kernel=2, pool_stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
    )


class CNNClassifier(nn.Module):
    def __init__(
        self,
        input_channels=IMAGE_SHAPE[0],  # 1
        channel_dims=(32, 64),
        hidden_dim=128,
        num_classes=NUM_CLASSES,
        dropout=0.2,
    ):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(input_channels, channel_dims[0]),    # [B, 1, 28, 28] -> [B, 32, 14, 14]
            conv_block(channel_dims[0], channel_dims[1]),   # [B, 32, 14, 14] -> [B, 64, 7, 7]
        )
        spatial = IMAGE_SHAPE[1] // 2 ** len(channel_dims)  # 28 / 4 = 7 after two conv_blocks

        # head (fully connected classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                   # [B, 64, 7, 7] -> [B, 3136]
            nn.Dropout(dropout),
            nn.Linear(channel_dims[-1] * spatial * spatial, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.softmax(self(x), dim=1)


MODELS = {
    "Softmax": SoftmaxClassifier,
    "MLP": MLPClassifier,
    "CNN": CNNClassifier,
}
