# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the new QuantumLayer module
from .quantum_layer import QuantumLayer

# Configuration for the QuantumLayer
# These values can be tuned.
QUANTUM_NUM_QUBITS = 4
QUANTUM_NUM_LAYERS_VQC = 3  # Number of layers in the StronglyEntanglingLayers ansatz


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class HybridDoubleConv(nn.Module):
    '''
    A hybrid classical-quantum convolutional block.
    It applies a classical convolution, then a QuantumLayer for feature mixing
    (per-pixel/channel-wise), and then another classical convolution.
    '''

    def __init__(self, in_ch, out_ch, num_qubits=QUANTUM_NUM_QUBITS, num_layers_vqc=QUANTUM_NUM_LAYERS_VQC):
        super(HybridDoubleConv, self).__init__()

        # First classical convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Quantum Layer
        # The QuantumLayer will process the 'out_ch' features at each spatial location.
        # Its input_features and output_features are set to 'out_ch' to maintain channel count.
        self.quantum_block = QuantumLayer(
            input_features=out_ch,
            output_features=out_ch,  # Quantum layer outputs same number of features
            num_qubits=num_qubits,
            num_layers_vqc=num_layers_vqc
        )

        # Second classical convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # First classical convolution
        x = self.conv1(x)  # Shape: (batch_size, out_ch, H, W)

        # Apply quantum layer
        # The QuantumLayer handles reshaping for per-pixel processing internally.
        # It takes (N, C, H, W) and effectively processes (N*H*W, C) through the QCircuit
        # then reshapes back to (N, C, H, W).
        x = self.quantum_block(x)

        # Second classical convolution
        x = self.conv2(x)  # Shape: (batch_size, out_ch, H, W)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        # Using the standard double_conv for the input block
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            # Replacing double_conv with HybridDoubleConv here
            # This means one of the downsampling blocks will have a quantum component.
            HybridDoubleConv(in_ch, out_ch,
                             num_qubits=QUANTUM_NUM_QUBITS,
                             num_layers_vqc=QUANTUM_NUM_LAYERS_VQC)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        # Using the standard double_conv for the upsampling block
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
