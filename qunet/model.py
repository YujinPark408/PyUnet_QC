# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch    

from .parts import *

class Q_UNet(nn.Module):
    """Quantum‑enhanced U‑Net.

    Restored the full encoder–decoder path (down4 / up4) so that the skip
    connections are consistent and the variable *x5* is always defined.
    """
    def __init__(self, input_channels, output_channels):
        super(Q_UNet, self).__init__()

        # Encoder ----------------------------------------------------------------
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        # Decoder ----------------------------------------------------------------
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, output_channels)

    def forward(self, x):
        # ---------------- Encoder ----------------
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # ---------------- Decoder ----------------
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x = self.up2(x4, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # x = self.outc(x)
        
        # return F.sigmoid(x)
        return torch.sigmoid(x)
