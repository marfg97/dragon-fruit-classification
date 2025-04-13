from torchvision import models
import torch.nn as nn

def get_resnet18(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def get_vgg16(num_classes, pretrained=True):
    model = models.vgg16(pretrained=pretrained)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    return model