import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo


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

        self.branch1 = nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1, dilation=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=2, dilation=2)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=4, dilation=4)
        )

        self.cbam = CBAM(in_channels, r)
        self.conv1 = nn.Conv2d(in_channels, output_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, output_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        cbam = self.cbam(x)
        out = out1 + out2 + out3 + self.conv1(cbam) + self.conv2(residual)
        return out

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

def SimAM(X, lamb):
    # spatial size
    n = X.shape[2] * X.shape[3]- 1
    # square of (t- u)
    d = (X- X.mean(dim=[2,3])).pow(2)
    # d.sum() / n is channel variance
    v = d.sum(dim=[2,3]) / n
    # E_inv groups all importance of X
    E_inv = d / (4 * (v + lamb)) + 0.5
    # return attended features
    return X * nn.Sigmoid(E_inv)

class ResSimAM(nn.Module):
    def __init__(self, in_channels, lamb):
        super(ResSimAM, self).__init__()

        self.lamb = lamb
        self.conv1 = ConvBNReLU(in_channels, in_channels, 3, 1, 1, 1)
        self.conv2 = ConvBNReLU(in_channels, in_channels, 3, 1, 1, 1, use_relu=False)

        self.simam = SimAM

    def forward(self, X):
        residual = X
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.simam(X, self.lamb)
        return nn.ReLU(X + residual)

class DCIM(nn.Module):
    def __init__(self, output_list, num_parallel, r=16):
        super(DCIM, self).__init__()
        self.levels = len(output_list)
        self.num_parallel = num_parallel

        # Initialize convolutional, upsampling, and downsampling layers
        self.H = nn.ModuleDict()
        self.D = nn.ModuleDict()
        self.U = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        for k in range(self.num_parallel):
            for level in range(self.levels):
                idx = f"{level}_{k}"
                input_channels = output_list[level] if level != 0 else 0 # Downsampled input channels
                input_channels += 0 if k == 0 or level == self.levels - 1 else output_list[level] + output_list[level+1] # Skip connection and Upsampling

                input_channels = output_list[level] if input_channels == 0 else input_channels

                # print(f"Level {l}, Branch {k}, Input channels: {input_channels}")

                self.H[idx] = ResHDCCBAM(input_channels, output_list[level], r)

                if level < self.levels - 1:
                    # self.D[idx] = DownSampling(input_channels, output_list[l+1])
                    self.D[idx] = DownSampling(output_list[level], output_list[level+1])
                # if k < self.num_parallel - 1:
                #     self.U[f"{l+1}_{k}"] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Initilize tensor storage
        X = {}

        for k in range(self.num_parallel):
            for lvl in range(self.levels):
                idx = f"{lvl}_{k}"

                if lvl == 0:
                    if k == 0:
                        X[idx] = self.H[idx](x)
                    else:
                        U = self.U(X[f"{lvl+1}_{k-1}"])
                        X[idx] = self.H[idx](torch.cat([X[f"{lvl}_{k-1}"], U], dim = 1))

                elif lvl > 0 and lvl < self.levels - 1:
                    if k == 0:
                        D = self.D[f"{lvl-1}_{k}"](X[f"{lvl-1}_{k}"])
                        X[idx] = self.H[idx](D)
                    else:
                        D = self.D[f"{lvl-1}_{k}"](X[f"{lvl-1}_{k}"])
                        U = self.U(X[f"{lvl+1}_{k-1}"])
                        X[idx] = self.H[idx](torch.cat([X[f"{lvl}_{k-1}"], D, U], dim = 1))
                elif lvl == self.levels - 1:
                    D = self.D[f"{lvl-1}_{k}"](X[f"{lvl-1}_{k}"])
                    X[idx] = self.H[idx](D)

        return [X[f"{lvl}_{self.num_parallel-1}"] for lvl in range(self.levels)]

class AFF(nn.Module):
    """
    Adaptive Feature Fusion (AFF) module for semantic segmentation
    """
    def __init__(self, in_channels, out_channels):
        super(AFF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x2 = nn.Upsample(size=x1.size()[2:], mode='bilinear', align_corners=True)(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class CARAFE(nn.Module):
    # CARAFE: Content-Aware ReAssembly of FEatures "https://arxiv.org/pdf/1905.02188.pdf"
    """
    Args:
        input_channels (int): input feature channels
        scale_factor (int): upsample ratio
        up_kernel (int): kernel size of CARAFE op
        up_group (int): group size of CARAFE op
        encoder_kernel (int): kernel size of content encoder
        encoder_dilation (int): dilation of content encoder

    Returns:
        upsampled feature map
    """
    def __init__(self, input_channels, scale_factor=2, kernel_up=5, kernel_encoder=3):
        super(CARAFE, self).__init__()
        self.scale_factor = scale_factor
        self.kernel_up = kernel_up
        self.kernel_encoder = kernel_encoder
        self.down = nn.Conv2d(input_channels, input_channels // 4, 1)
        self.encoder = nn.Conv2d(input_channels // 4, self.scale_factor ** 2 * self.kernel_up ** 2,self.kernel_encoder, 1, self.kernel_encoder // 2)
        self.out = nn.Conv2d(input_channels, input_channels, 1)

    def forward(self, x):
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.scale_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.scale_factor, step=self.scale_factor) # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.scale_factor, step=self.scale_factor) # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_up ** 2, H, W, self.scale_factor ** 2) # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(x, pad=(self.kernel_up // 2, self.kernel_up // 2,
                                          self.kernel_up // 2, self.kernel_up // 2),
                          mode='constant', value=0) # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_up, step=1) # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_up, step=1) # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1) # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.scale_factor)

        out_tensor = self.out(out_tensor)
        #print("up shape:",out_tensor.shape)
        return out_tensor

class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class AdaptiveThresholdPrediction(nn.Module):
    def __init__(self, in_channels=64, pool_size=(1, 1), num_classes=1, initial_threshold=0.5):
        """
        Adaptive Threshold Prediction Module.
        Args:
            in_channels (int): Number of input channels.
            intermediate_channels (int): Number of intermediate channels (e.g., 64).
            pool_size (tuple): Adaptive pooling output size (default 1x1).
        returns:
            torch.Tensor: Thresholded output tensor of the same shape as the input tensor.
        """
        super(AdaptiveThresholdPrediction, self).__init__()

        # First path: Conv1x1 -> STAF
        self.conv1x1_main = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)

        # STAF branch
        self.staf_branch = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Second path: Adaptive Pooling -> Conv1x1 -> Sigmoid -> ATS
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=pool_size)
        self.ats_branch = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # Learnable scalar for threshold prediction
        self.threshold = nn.Parameter(torch.tensor(initial_threshold, requires_grad=True))

    def forward(self, x):
        """
        Forward pass of the Adaptive Threshold Prediction Module.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Thresholded output tensor.
        """
        # First path
        main_output = self.conv1x1_main(x)  # Output: (B, 64, H, W)

        # STAF branch
        staf_output = self.staf_branch(main_output)  # Output: (B, num_classes, H, W)

        # Second path
        pooled = self.adaptive_pool(x)  # Output: (B, C, 1, 1)
        ats_output = self.ats_branch(pooled)  # Output: (B, num_classes, 1, 1)

        # Adaptive thresholding
        scaled_threshold = self.threshold * ats_output  # Output: (B, num_classes, 1, 1)

        scaled_threshold = F.interpolate(scaled_threshold, size=main_output.shape[2:], mode='bilinear', align_corners=False)

        # Thresholding
        output = staf_output * scaled_threshold  # Output: (B, num_classes, H, W)

        return output


class WResHDC_FF(nn.Module):
    def __init__(self, num_classes, input_channels, output_list, num_parallel, channel_ratio=16, upsample_cfg=dict(type='carafe', scale_factor = 2, kernel_up = 5, kernel_encoder = 3, compress_channels = 64)):
        """
        The overall architecture of the WResHDC-FF models for semantic segmentation.
        It contains the following components:
            1. ResHDCCBAM: Residual Hierarchical Dense Convolutional Channel Attention Block
            2. AFF: Adaptive Feature Fusion
            3. CARAFE: Content-Aware ReAssembly of FEatures

        Args:
            num_classes (int): Number of classes for the segmentation task
            output_list (list): List of output channels for each level
            num_parallel (int): Number of parallel branches in each level
            r (int): Reduction ratio for the channel attention module
            upsample_cfg (dict): Configuration for the upsampling module
        """
        super(WResHDC_FF, self).__init__()

        self.levels = len(output_list) # 5
        self.num_parallel = num_parallel # 1
        self.channel_ratio = channel_ratio

        # First convolutional layer as compress layer
        # self.conv1 = ConvBNReLU(input_channels, output_list[0], 3, 1, 1, 1)
        self.conv1 = nn.Conv2d(input_channels, output_list[0], kernel_size=1, stride=1)

        # Initialize DCIM module
        self.dcim = DCIM(output_list, num_parallel)

        self.aff = nn.ModuleList()
        # Initialize AFF module
        for level in range(self.levels - 2):
            self.aff.append(AFF(output_list[level] + output_list[level+1], output_list[level]))

        # Initialize CARAFE module
        self.upsample = nn.ModuleList()
        self.convbnrelu1 = nn.ModuleList()
        self.convbnrelu2 = nn.ModuleList()
        for level in range(self.levels - 1, 0, -1):
            if upsample_cfg['type'] == 'carafe':
                self.upsample.append(CARAFE(output_list[level],
                                            upsample_cfg['scale_factor'],
                                            upsample_cfg['kernel_up'],
                                            upsample_cfg['kernel_encoder']))
            else:
                self.upsample.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

            self.convbnrelu1.append(ConvBNReLU(output_list[level] + output_list[level-1], output_list[level], 3, 1, 1, 1))
            self.convbnrelu2.append(ConvBNReLU(output_list[level], output_list[level-1], 3, 1, 1, 1))

        # Adaptive Threshold Prediction Module
        self.atp = AdaptiveThresholdPrediction(in_channels=output_list[0], pool_size=(1, 1), num_classes=num_classes)

    def forward(self, x):

        x = self.conv1(x)  # 3*320*320 -> 64*320*320
        # DCIM module
        X = self.dcim(x) # levels of feature maps
        F = X.copy()
        # delete two last levels
        del F[self.levels-1]

        # Adaptive Feature Fusion module
        for level in range(self.levels - 2): # 0, 1, 2
            F[level] = self.aff[level](X[level], X[level+1])

        # Upsampling module
        # out = self.upsample[0](X[self.levels - 1]) # W*H*1024 -> 2W*2H*1024
        # out = torch.cat([out, X[self.levels - 2]], dim=1)
        # out = self.convbnrelu1[0](out)
        # out = self.convbnrelu2[0](out) # 2W*2H*1024 -> 2W*2H*512

        # out = self.upsample[1](out) # 2W*2H*512 -> 4W*4H*512
        # out = torch.cat([out, F[self.levels - 3]], dim=1)
        # out = self.convbnrelu1[1](out)
        # out = self.convbnrelu2[1](out) # 4W*4H*512 -> 4W*4H*256

        # out = self.upsample[2](out)
        # out = torch.cat([out, F[self.levels - 4]], dim=1)
        # out = self.convbnrelu1[2](out)
        # out = self.convbnrelu2[2](out)

        # out = self.upsample[3](out)
        # out = torch.cat([out, F[self.levels - 5]], dim=1)
        # out = self.convbnrelu1[3](out)
        # out = self.convbnrelu2[3](out)

        out = X[self.levels - 1]
        for level in range(self.levels - 1, 0, -1):
            out = self.upsample[self.levels - 1 - level](out)
            out = torch.cat([out, F[level-1]], dim=1)
            out = self.convbnrelu1[self.levels - 1 - level](out)
            out = self.convbnrelu2[self.levels - 1 - level](out)

        return self.atp(out)

class ResHDC_Model(nn.Module):
    def __init__(self, num_classes, input_channels, output_list, channel_ratio=16, upsample_cfg=dict(type='carafe', scale_factor = 2, kernel_up = 5, kernel_encoder = 3, compress_channels = 64)):
        super(ResHDC_Model, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(input_channels, output_list[0], kernel_size=1, stride=1) # 3*320*320 -> 64*320*320
        
        self.reshdc_branch = nn.ModuleList()
        for i in range(len(output_list)-1):
            self.reshdc_branch.append(ResHDCCBAM(output_list[i], output_list[i+1], channel_ratio))

        # adaptive feature fusion = len(output_list) - 2
        self.aff = nn.ModuleList()
        for i in range(len(output_list) - 2):
            self.aff.append(AFF(output_list[i] + output_list[i+1], output_list[i]))

        # upsampling with CARAFE = len(output_list) - 1
        self.upsample = nn.ModuleList()
        self.convbnrelu1 = nn.ModuleList()
        self.convbnrelu2 = nn.ModuleList()

        for i in range(len(output_list) - 1):
            if upsample_cfg['type'] == 'carafe':
                self.upsample.append(CARAFE(output_list[i], upsample_cfg['scale_factor'], upsample_cfg['kernel_up'], upsample_cfg['kernel_encoder']))
            else:
                self.upsample.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

            self.convbnrelu1.append(ConvBNReLU(output_list[i] + output_list[i+1], output_list[i], 3, 1, 1, 1))
            self.convbnrelu2.append(ConvBNReLU(output_list[i], output_list[i+1], 3, 1, 1, 1))

        # Adaptive Threshold Prediction Module
        self.atp = AdaptiveThresholdPrediction(in_channels=output_list[0], pool_size=(1, 1), num_classes=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        X = []
        for i in range(len(self.reshdc_branch)):
            x = self.reshdc_branch[i](x)
            X.append(x)

        F = X.copy()
        del F[-1]
        for i in range(len(self.aff)):
            F[i] = self.aff[i](X[i], X[i+1])

        out = X[-1]
        for i in range(len(self.upsample)):
            out = self.upsample[i](out)
            out = torch.cat([out, F[i]], dim=1)
            out = self.convbnrelu1[i](out)
            out = self.convbnrelu2[i](out)

        return self.atp(out)

class UNetEncoder(nn.Module):
    def __init__(self, input_channels, output_list):
        super(UNetEncoder, self).__init__()

        self.conv1 = ConvBNReLU(input_channels, output_list[0], kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = ConvBNReLU(output_list[0], output_list[0], kernel_size=3, stride=1, padding=1, dilation=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBNReLU(output_list[0], output_list[1], kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv4 = ConvBNReLU(output_list[1], output_list[1], kernel_size=3, stride=1, padding=1, dilation=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBNReLU(output_list[1], output_list[2], kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv6 = ConvBNReLU(output_list[2], output_list[2], kernel_size=3, stride=1, padding=1, dilation=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = ConvBNReLU(output_list[2], output_list[3], kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv8 = ConvBNReLU(output_list[3], output_list[3], kernel_size=3, stride=1, padding=1, dilation=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = ConvBNReLU(output_list[3], output_list[4], kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv10 = ConvBNReLU(output_list[4], output_list[4], kernel_size=3, stride=1, padding=1, dilation=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, inputs):
        x1 = self.conv1(inputs)
        x1 = self.conv2(x1)
        x2 = self.pool1(x1)

        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        x3 = self.pool2(x2)

        x3 = self.conv5(x3)
        x3 = self.conv6(x3)
        x4 = self.pool3(x3)

        x4 = self.conv7(x4)
        x4 = self.conv8(x4)
        x5 = self.pool4(x4)

        x5 = self.conv9(x5)
        x5 = self.conv10(x5)
        x6 = self.pool5(x5)

        return x1, x2, x3, x4, x5, x6

class UNetDecoder(nn.Module):
    def __init__(self, output_list, num_classes):
        super(UNetDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv11 = ConvBNReLU(output_list[4]*2, output_list[3], kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv12 = ConvBNReLU(output_list[3], output_list[3], kernel_size=3, stride=1, padding=1, dilation=1)

        self.conv13 = ConvBNReLU(output_list[3]*2 + output_list[4], output_list[2], kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv14 = ConvBNReLU(output_list[2], output_list[2], kernel_size=3, stride=1, padding=1, dilation=1)

        self.conv15 = ConvBNReLU(output_list[2]*2 + output_list[3], output_list[1], kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv16 = ConvBNReLU(output_list[1], output_list[1], kernel_size=3, stride=1, padding=1, dilation=1)

        self.conv17 = ConvBNReLU(output_list[1]*2 + output_list[2], output_list[0], kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv18 = ConvBNReLU(output_list[0], output_list[0], kernel_size=3, stride=1, padding=1, dilation=1)

        self.conv19 = ConvBNReLU(output_list[0]*2 + output_list[1], num_classes, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=False)
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x1, x2, x3, x4, x5, x6):
        out = self.upsample(x6)
        out = torch.cat([out, x5], dim=1)
        out = self.conv11(out)
        out = self.conv12(out)

        out = self.upsample(out)
        x5 = self.upsample(x5)
        out = torch.cat([out, x4, x5], dim=1)
        out = self.conv13(out)
        out = self.conv14(out)

        out = self.upsample(out)
        x4 = self.upsample(x4)
        out = torch.cat([out, x3, x4], dim=1)
        out = self.conv15(out)
        out = self.conv16(out)

        out = self.upsample(out)
        x3 = self.upsample(x3)
        out = torch.cat([out, x2, x3], dim=1)
        out = self.conv17(out)
        out = self.conv18(out)

        out = self.upsample(out)
        x2 = self.upsample(x2)
        out = torch.cat([out, x1, x2], dim=1)
        out = self.conv19(out)
        out = self.sigmoid(out)

        return out

class UNet(nn.Module):
    def __init__(self, num_classes, input_channels, output_list):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.output_list = output_list

        self.encoder = UNetEncoder(input_channels, output_list)
        self.decoder = UNetDecoder(output_list, num_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.encoder(x)
        out = self.decoder(x1, x2, x3, x4, x5, x6)
        return out

# Example usage
if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 320, 320)
    output_list = [32, 64, 128, 256, 512]
    num_parallel = 2
    num_classes = 1
    upsampling_cfg = dict(type='carafe', scale_factor=2, kernel_up=5, kernel_encoder=3)

    # model = WResHDC_FF(num_classes, input_tensor.shape[1], output_list, num_parallel, upsampling_cfg)
    # model = DCIM(output_list, num_parallel)
    # model = UNet(num_classes, input_tensor.shape[1], output_list)
    model = ResHDC_Model(num_classes, input_tensor.shape[1], output_list)
    output = model(input_tensor)

    # print("Output shape:", output.shape)
    # print("Output shape:", [o.shape for o in output])

    # print model summary
    model_info = torchinfo.summary(model, input_size=(1, 3, 320, 640))
    print(model_info)
