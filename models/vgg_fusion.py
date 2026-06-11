import torch
import torch.nn as nn
import torchvision.models as models

# ==============================================================================
# VGG BASELINE (Evaluates individual effects of background preprocessing variants)
# ==============================================================================

def vgg16_single_stream(num_classes=2, pretrained=True):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
    
    # Freeze feature extraction tracking blocks
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace final linear node mapping categories
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model

# ==============================================================================
# VGG16 EARLY FUSION ARCHITECTURE (Channel-level Splice)
# ==============================================================================

def vgg16_early_fusion(num_classes=2, in_channels=6, pretrained=True):
    """
    Factory function adapting VGG16 for Early Fusion.
    Slices the entry layer container to accommodate multi-stream input feature stacks.
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)

    # Reconstruct the first Conv layer
    original_conv1 = model.features[0]
    new_conv1 = nn.Conv2d(
        in_channels=in_channels, 
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size, 
        stride=original_conv1.stride, 
        padding=original_conv1.padding
    )

    if pretrained:
        # Load standard ImageNet weights into channels 0, 1, 2
        new_conv1.weight.data[:, :3, :, :] = original_conv1.weight.data
        new_conv1.bias.data = original_conv1.bias.data
        nn.init.kaiming_normal_(new_conv1.weight.data[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
        print("Pretrained VGG16 features adapted for multi-channel input streaming.")

    model.features[0] = new_conv1

    # Freeze feature extractors except for the new front layer
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[0].parameters():
        param.requires_grad = True

    # Remap top classification layers to map binary categories
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model


# ==============================================================================
# VGG16 LATE FUSION ARCHITECTURE (Feature-level Concat)
# ==============================================================================

class LateFusionVGG(nn.Module):
    """
    Dual-branch VGG16 Late Fusion model.
    Flattens spatial dimension outputs at the dense terminal and feeds concatenated features.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None

        # Build dual streams targeting feature space extraction
        self.vgg_branch1 = models.vgg16(weights=weights).features
        self.vgg_branch2 = models.vgg16(weights=weights).features

        # Freeze trunk param updates
        for param in self.vgg_branch1.parameters():
            param.requires_grad = False
        for param in self.vgg_branch2.parameters():
            param.requires_grad = False

        # Output feature size: (512 channels * 7 * 7 spatial grid) * 2 streams = 50176 dims
        fused_features_size = (512 * 7 * 7) * 2

        self.classifier = nn.Sequential(
            nn.Linear(fused_features_size, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x1, x2):
        x1 = self.vgg_branch1(x1)
        x2 = self.vgg_branch2(x2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        fused_features = torch.cat((x1, x2), dim=1)
        return self.classifier(fused_features)