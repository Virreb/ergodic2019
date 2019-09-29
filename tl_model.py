

def get_resnet18_model():
    from torchvision import models
    from config import device
    import torch.nn as nn

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)

    model.to(device)

    return model
