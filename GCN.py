import torch.nn.functional as func
from torch import nn
from torchvision import models


# Inspiration from https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/models/gcn.py
class _GlobalConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(_GlobalConvolutionalNetwork, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv_l1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0))
        self.conv_l2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, pad))
        self.conv_r1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, pad))
        self.conv_r2 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        out = x_l + x_r
        return out


class _BoundaryRefinementBlock(nn.Module):
    def __init__(self, n_channels):
        super(_BoundaryRefinementBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)  # Read about inplace
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)  # pad = 1 if (height % 2 == 0 and width % 2 == 0)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class GCN(nn.Module):
    def __init__(self, n_classes, pretrained_resnet=True):
        super(GCN, self).__init__()
        #self.input_size = input_size
        resnet = models.resnet50(pretrained=pretrained_resnet)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcn1 = _GlobalConvolutionalNetwork(2048, n_classes, 15)
        self.gcn2 = _GlobalConvolutionalNetwork(1024, n_classes, 15)
        self.gcn3 = _GlobalConvolutionalNetwork(512, n_classes, 15)
        self.gcn4 = _GlobalConvolutionalNetwork(256, n_classes, 15)

        self.brb1 = _BoundaryRefinementBlock(n_classes)
        self.brb2 = _BoundaryRefinementBlock(n_classes)
        self.brb3 = _BoundaryRefinementBlock(n_classes)
        self.brb4 = _BoundaryRefinementBlock(n_classes)
        self.brb5 = _BoundaryRefinementBlock(n_classes)
        self.brb6 = _BoundaryRefinementBlock(n_classes)
        self.brb7 = _BoundaryRefinementBlock(n_classes)
        self.brb8 = _BoundaryRefinementBlock(n_classes)
        self.brb9 = _BoundaryRefinementBlock(n_classes)

    def forward(self, x):
        # if x: 512
        fm0 = self.layer0(x)  # 256 - feature map 0
        fm1 = self.layer1(fm0)  # 128
        fm2 = self.layer2(fm1)  # 64
        fm3 = self.layer3(fm2)  # 32
        fm4 = self.layer4(fm3)  # 16

        gcfm1 = self.brb1(self.gcn1(fm4))  # 16
        gcfm2 = self.brb2(self.gcn2(fm3))  # 32
        gcfm3 = self.brb3(self.gcn3(fm2))  # 64
        gcfm4 = self.brb4(self.gcn4(fm1))  # 128

        fs1 = self.brb5(func.interpolate(gcfm1, fm3.size()[2:], mode='bilinear', align_corners=True) + gcfm2)  # 32
        fs2 = self.brb6(func.interpolate(fs1, fm2.size()[2:], mode='bilinear', align_corners=True) + gcfm3)  # 64
        fs3 = self.brb7(func.interpolate(fs2, fm1.size()[2:], mode='bilinear', align_corners=True) + gcfm4)  # 128
        fs4 = self.brb8(func.interpolate(fs3, fm0.size()[2:], mode='bilinear', align_corners=True))  # 256
        out = self.brb9(func.interpolate(fs4, x.shape[2:], mode='bilinear', align_corners=True))  # 512
        return out


if __name__=='__main__':
    import torch
    import numpy as np
    in_arr = torch.from_numpy(np.random.randn(1, 3, 512, 512)).float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from torchsummary import summary
    #net = UNet(3, 4).float()
    net = GCN(4)
    net = net.to(device)
    in_arr = in_arr.to(device)
    print('loaded custom')

    for tmp_layer in [net.layer2, net.layer3, net.layer4]:
        for param in tmp_layer.parameters():
            param.requires_grad = False
    in_shape_x, in_shape_y = 512, 512
    print(summary(net, (3, in_shape_x, in_shape_y)))
    print('-----------')
    print('-----------')
    """
    print('-----------')
    print('-----------')
    net = UNet(3, 4).float()
    net = net.to(device)
    in_arr = in_arr.to(device)
    print('loaded custom')
    #for param in net.resnet.parameters():
    #    param.requires_grad = False
    in_shape_x, in_shape_y = 512, 512
    print(summary(net, (3, in_shape_x, in_shape_y)))
    """