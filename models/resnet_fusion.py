import torch
import torch.nn as nn
import torchvision.models as models

# ==============================================================================
# RESNET BASELINE (Evaluates individual effects of background preprocessing variants)
# ==============================================================================

def resnet18_single_stream(num_classes=2, pretrained=True):
    """
    Baseline 1: Standard Single-Stream ResNet18 (No Fusion).
    Evaluates individual effects of background preprocessing variants.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    
    # Freeze backbone parameters for standard transfer learning fine-tuning
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final fully-connected classifier head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ==============================================================================
# RESNET EARLY FUSION COMPONENTS (Custom Sub-architecture)
# ==============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # Directed input from conv1 output 'out'
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetCustom(nn.Module):
    """
    Custom ResNet base configured to adapt input channel dimensionality 
    for multi-stream Early Fusion.
    """
    def __init__(self, block, layers, num_classes=1000, in_channels=3, include_top=True, dropout_rate=0.5):
        super(ResNetCustom, self).__init__()
        self.include_top = include_top
        self.in_channels = 64

        # Modified first layer to accept arbitrary input channels (e.g., 6 channels)
        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
        return x


def resnet18_early_fusion(num_classes=2, in_channels=6, pretrained=True):
    """
    Factory function for 6-channel Early Fusion ResNet18.
    Splices pretrained ImageNet weights into the first 3 entry channels.
    """
    model_orig = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model = ResNetCustom(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)

    if pretrained:
        state_dict_orig = model_orig.state_dict()
        state_dict_new = model.state_dict()

        for name, param in state_dict_orig.items():
            if name not in state_dict_new:
                continue
            if name == 'conv1.weight':
                state_dict_new['conv1.weight'][:, :3, :, :] = param
                nn.init.kaiming_normal_(state_dict_new['conv1.weight'][:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            elif name in ['fc.weight', 'fc.bias']:
                continue
            else:
                state_dict_new[name] = param

        model.load_state_dict(state_dict_new)
        print("Pretrained ResNet18 weights mapped into 6-channel Early Fusion backbone.")

    return model


# ==============================================================================
# RESNET LATE FUSION ARCHITECTURE (Feature-level Splice)
# ==============================================================================

class LateFusionResNet(nn.Module):
    """
    Dual-branch ResNet18 Late Fusion model.
    Extracts deep features from separate inputs and maps them into a joint classifier.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None

        self.resnet_branch1 = models.resnet18(weights=weights)
        self.resnet_branch2 = models.resnet18(weights=weights)

        # Sever original fully connected layers
        self.resnet_branch1.fc = nn.Identity()
        self.resnet_branch2.fc = nn.Identity()

        # Freeze backbones
        for param in self.resnet_branch1.parameters():
            param.requires_grad = False
        for param in self.resnet_branch2.parameters():
            param.requires_grad = False

        # Concat size: 512 + 512 = 1024
        fused_features_size = 512 * 2

        self.classifier = nn.Sequential(
            nn.Linear(fused_features_size, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x1, x2):
        x1 = self.resnet_branch1(x1)
        x2 = self.resnet_branch2(x2)
        fused_features = torch.cat((x1, x2), dim=1)
        return self.classifier(fused_features)