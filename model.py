import normalizations
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, image_dim, num_classes, norm_type, use_norm=False, **kwargs):
        super().__init__()
        hook = normalizations.NORMALIZATIONS[norm_type](image_dim, **kwargs)
        self.layer1 = nn.Sequential(
            *hook.conv_hook(
                nn.Conv2d(
                    1, kwargs["l1_size"], kwargs["kernel_size"], stride=1, padding=2
                ),
                norm=use_norm,
            ),
            nn.ReLU(),
            *hook.conv_hook(nn.MaxPool2d(kernel_size=2, stride=2))
        )
        self.layer2 = nn.Sequential(
            *hook.conv_hook(
                nn.Conv2d(
                    kwargs["l1_size"],
                    kwargs["l2_size"],
                    kwargs["kernel_size"],
                    stride=1,
                    padding=2,
                ),
                norm=use_norm,
            ),
            nn.ReLU(),
            *hook.conv_hook(nn.MaxPool2d(kernel_size=2, stride=2))
        )
        self.fc = nn.Linear(7 * 7 * kwargs["l2_size"], num_classes)
        self.softmax = nn.Softmax(num_classes)

    def forward(self, inp):
        out = self.layer1(inp)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
