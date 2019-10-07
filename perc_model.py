import torch.nn as nn
from torchvision import models
from config import device


def transfer_learning(model, nbr_target_classes):
    for param in model.parameters():
        param.requires_grad = False

    fc_size = 128
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
               nn.Linear(num_features, fc_size),    # 128 fully connected before last
               nn.ReLU(inplace=True),
               nn.Linear(fc_size, nbr_target_classes)).to(device)
    # model.fc = nn.Linear(num_features, nbr_target_classes)
    model.to(device)

    return model


def get_resnet_18(nbr_classes):
    return transfer_learning(models.resnet18(pretrained=True), nbr_classes)


def get_resnet_101(nbr_classes):
    return transfer_learning(models.resnet101(pretrained=True), nbr_classes)


def get_deep_lab_v3(nbr_classes):
    return transfer_learning(models.segmentation.deeplabv3_resnet101(pretrained=True), nbr_classes)
