import torch.nn.functional as F
from torch import nn
from torchvision import models
import torch


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class DeeplabFork(nn.Module):
    def __init__(self, pretrained=True, freezed_backbone=True, freezed_aspp=False):
        super(DeeplabFork, self).__init__()
        deeplab = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        self.backbone = deeplab.backbone
        self.aspp = deeplab.classifier._modules['0']
        if freezed_backbone:
            for param in deeplab.backbone.parameters():
                param.requires_grad = False
        if freezed_aspp:
            for param in self.aspp.parameters():
                param.requires_grad = False

        num_classes = 4
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        x = features["out"]
        x = self.aspp(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        soft_max_activation = F.softmax(x, dim=1)
        class_sum = torch.sum(soft_max_activation, dim=2)
        class_sum = torch.sum(class_sum, dim=2)
        class_fraction = class_sum/(soft_max_activation.shape[2] * soft_max_activation.shape[3])

        return x, class_fraction


if __name__=='__main__':
    import torch
    import numpy as np
    in_arr = torch.from_numpy(np.random.randn(2, 3, 512, 512)).float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from torchsummary import summary
    in_shape_x, in_shape_y = 512, 512

    deeplab = DeeplabFork()



    model_parameters = filter(lambda p: p.requires_grad, deeplab.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Trainable params')
    print(params)

    model_parameters = filter(lambda p: not p.requires_grad, deeplab.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Non trainable params')
    print(params)


