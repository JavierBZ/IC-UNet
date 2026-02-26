from monai.networks.nets import UNet, AutoEncoder, SwinUNETR, UNETR

import torch
import torch.nn as nn
import torch.nn.functional as F

def neural_network(config):
    """
    Returns the neural network model based on the configuration.

    Args:
        config (dict): Configuration dictionary containing 'model', 'classes', and 'resnet_layers' keys.

    Returns:
        torch.nn.Module: Neural network model.
    """
    if config["model"] == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=config["channels"],
            out_channels=config["classes"],
            channels=(16, 32, 64, 128, 256, 512),
            dropout=config["dropout"],
            strides=(2, 2, 2, 2, 2),
            num_res_units=config["resnet_layers"],
        ).cuda()
    elif config["model"] == "autoencoder":
        return AutoEncoder(
            spatial_dims=3,
            in_channels=config["channels"],
            out_channels=config["classes"],
            channels=(2, 4, 8),
            strides=(2, 2, 2),
        ).cuda()
    elif config["model"] == "unetr":
        return UNETR(
            img_size=(256, 256, 128),
            spatial_dims=3,
            in_channels=config["channels"],
            out_channels=config["classes"],
            proj_type="conv",
            mlp_dim=2048,
            feature_size=8,
            hidden_size=256,
            num_heads=8,
            dropout_rate=config["dropout"],
            norm_name="instance",
        ).cuda()
    elif config["model"] == "SwinUNETR":
        return SwinUNETR(
            img_size=(256, 256, 128),
            spatial_dims=3,
            in_channels=config["channels"],
            out_channels=config["classes"],
            depths=(2, 2, 2),
            use_v2=True,
        ).cuda()
    
    elif config["model"] == "CSNet":
        return CSNet3D(
            classes=config["classes"],
            channels=config["channels"]
        ).cuda()
    
    else:
        raise ValueError("Invalid model specified in config.")

# # Enable anomaly detection for debugging in-place operations
# torch.autograd.set_detect_anomaly(True)

def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResEncoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)  # Ensure no in-place operation here
        return out


class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv(x)


class SpatialAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock3d, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.judge = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W, D = x.size()
        proj_query = self.query(x).view(B, -1, W * H * D).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H * D)
        proj_judge = self.judge(x).view(B, -1, W * H * D).permute(0, 2, 1)

        affinity1 = torch.matmul(proj_query, proj_key)
        affinity2 = torch.matmul(proj_judge, proj_key)
        affinity = torch.matmul(affinity1, affinity2)
        affinity = self.softmax(affinity)

        proj_value = self.value(x).view(B, -1, H * W * D)
        weights = torch.matmul(proj_value, affinity)
        weights = weights.view(B, C, H, W, D)
        return self.gamma * weights + x


class ChannelAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock3d, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W, D = x.size()
        proj_query = x.view(B, C, -1).permute(0, 2, 1)
        proj_key = x.view(B, C, -1)
        proj_judge = x.view(B, C, -1).permute(0, 2, 1)
        affinity1 = torch.matmul(proj_key, proj_query)
        affinity2 = torch.matmul(proj_key, proj_judge)
        affinity = torch.matmul(affinity1, affinity2)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W, D)
        return self.gamma * weights + x


class AffinityAttention3d(nn.Module):
    def __init__(self, in_channels):
        super(AffinityAttention3d, self).__init__()
        self.sab = SpatialAttentionBlock3d(in_channels)
        self.cab = ChannelAttentionBlock3d(in_channels)

    def forward(self, x):
        sab = self.sab(x)
        cab = self.cab(x)
        return sab + cab + x


class CSNet3D(nn.Module):
    def __init__(self, classes, channels):
        super(CSNet3D, self).__init__()
        self.enc_input = ResEncoder3d(channels, 16)
        self.encoder1 = ResEncoder3d(16, 32)
        self.encoder2 = ResEncoder3d(32, 64)
        self.encoder3 = ResEncoder3d(64, 128)
        self.encoder4 = ResEncoder3d(128, 256)
        self.downsample = downsample()
        self.affinity_attention = AffinityAttention3d(256)
        self.attention_fuse = nn.Conv3d(256 * 2, 256, kernel_size=1)
        self.decoder4 = Decoder3d(256, 128)
        self.decoder3 = Decoder3d(128, 64)
        self.decoder2 = Decoder3d(64, 32)
        self.decoder1 = Decoder3d(32, 16)
        self.deconv4 = deconv(256, 128)
        self.deconv3 = deconv(128, 64)
        self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(32, 16)
        self.final = nn.Conv3d(16, classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        input_feature = self.encoder4(down4)
        attention = self.affinity_attention(input_feature)
        attention_fuse = input_feature + attention

        up4 = self.deconv4(attention_fuse)
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        # final = torch.sigmoid(final)
        return final