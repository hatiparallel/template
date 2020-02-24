import numpy as np
import torch
import torchvision.models



class VGGUncertainty(nn.Module):
    def __init__(self, clip_features, kind = '16'):
        super(VGGUncertainty, self).__init__()
        self.vgg = call_partial_vgg(clip_features, kind)
        self.output = nn.Linear(4096, 1)
        self.uncertainty = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.vgg(x)
        y = self.output(x)
        u = self.uncertainty(x)

        return y, u

def call_partial_vgg(clip_features, kind):
    model = torchvision.models.vgg16(pretrained=True)
    if kind == '19':
        model = torchvision.models.vgg19(pretrained=True)
    elif kind == 'bn16':
        model = torchvision.models.vgg16_bn(pretrained=True)
    elif kind == 'bn19':
        model = torchvision.models.vgg19_bn(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(512*7*7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
    )

    if clip_features:
        for p in model.features.parameters():
            p.requires_grad = False
            
    return model



class ResNet(nn.Module):
    def __init__(self, layers):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__[
            'resnet{}'.format(layers)](pretrained=pretrained)

        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # pretrained_model._modules['avgpool']
        self.output = nn.Linear(512 * 4, 1)

        # clear memory
        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.output(x)

        return y
