import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        groups: int = 1,
        stride: int = 1,
        activation: bool = True,
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ] + ([nn.ReLU6(inplace=True)] if activation else [])
        super().__init__(*layers)


class CRPBlock(nn.Module):
    """
    Chained Residual Pooling
    Reference: https://ar5iv.org/html/1809.04766
    """

    def __init__(self, in_channels, out_channels, num_stages=1, use_groups=False):
        super().__init__()
        groups = in_channels if use_groups else 1
        convs = [
            nn.Conv2d(
                in_channels if (i == 0) else out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                groups=groups,
            )
            for i in range(num_stages)
        ]
        self.convs = nn.ModuleList(convs)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        out = x
        for conv in self.convs:
            out = conv(self.pool(out))
            x = out + x
        return x


class UnetBlock(nn.Module):
    def __init__(
        self,
        in_up,
        in_side,
        out_channels,
        kernel_size=1,
        num_stages=4,
        use_groups=False,
    ):
        super().__init__()
        self.conv_up = ConvLayer(in_up, out_channels, kernel_size)
        self.conv_side = ConvLayer(in_side, out_channels, kernel_size)
        self.crp = CRPBlock(
            out_channels, out_channels, num_stages=num_stages, use_groups=use_groups
        )

    def forward(self, side_input, up_input):
        up_input = self.conv_up(up_input)
        side_input = self.conv_side(side_input)
        if up_input.shape[-2:] != side_input.shape[-2:]:
            up_input = F.interpolate(
                up_input,
                size=side_input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        out = self.crp(F.relu(up_input + side_input))
        return out


class DynamicUnet(nn.Module):
    """
    A Unet that take almost any backbone from timm
    Reference: https://github.com/tcapelle/hydra_net/blob/master/hydranet/models.py#L13
    """

    def __init__(self, backbone="mobilenetv2_100", dim=256):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, features_only=True)
        # passing dummy tensor to get sizes
        dummy_tensor = torch.rand([1, 3, 64, 64])
        features = self.encoder(dummy_tensor)
        ch_sizes = [list(f.shape)[1] for f in features][::-1]
        self.upsample_blocks = nn.ModuleList()
        self.mid_conv = ConvLayer(ch_sizes[0], dim, 3)
        for i, ch_size in enumerate(ch_sizes[1:]):
            self.upsample_blocks.append(
                UnetBlock(
                    dim,
                    ch_size,
                    out_channels=dim,
                    use_groups=(i == (len(features) - 2)),
                )
            )

    def forward(self, x):
        input_shape = x.shape
        # features reversed in order
        features = self.encoder(x)[::-1]
        # put last feature on dim of the model
        x = self.mid_conv(features[0])
        # upsample blocks with shortcurts from the sides (jit friendly)
        for idx, ublock in enumerate(self.upsample_blocks):
            x = ublock(features[1:][idx], x)
        x = F.interpolate(
            x, size=input_shape[-2:], mode="bilinear", align_corners=False
        )
        return x


class SegmentationModel(nn.Module):
    def __init__(self, backbone="mobilenetv2_100", hidden_dim=256, num_classes=21):
        super().__init__()
        self.backbone = DynamicUnet(backbone, dim=hidden_dim)
        self.segmentation_head = nn.Sequential(
            ConvLayer(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        backbone_out = self.backbone(x)
        return self.segmentation_head(backbone_out)
