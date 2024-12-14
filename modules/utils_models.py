import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x
        return output

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output

class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output

class ResHDCCBAM(nn.Module):
    def __init__(self, in_channels, output_channels, r=16):
        super(ResHDCCBAM, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4)
        )

        self.cbam = CBAM(in_channels, r)


    def forward(self, x):
        residual = x
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        cbam = self.cbam(x)
        out = out1 + out2 + out3 + cbam + residual
        return out

class DCIM(nn.Module):
    def __init__(self, input_channels, output_list, r=16):
        pass

# Example usage
if __name__ == '__main__':
    input_tensor = torch.randn(1, 512, 40, 40)
    model = ResHDCCBAM(512)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should be [1, 512, 40, 40]
