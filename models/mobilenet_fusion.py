import torch
import torch.nn as nn
import torchvision.models as models

# ==============================================================================
# MOBILENETV2 EARLY FUSION ARCHITECTURE (Channel-level Concat)
# ==============================================================================

def mobilenet_v2_early_fusion(num_classes=2, in_channels=6, pretrained=True):
    """
    Factory function adapting MobileNetV2 for Early Fusion.
    Modifies the core entry layer 'features[0][0]' to support 6-channel streaming,
    retaining original pretrained weights in the first 3 channels.
    """
    # 1. Load standard model for pretrained source mapping
    model_orig = models.mobilenet_v2(pretrained=pretrained)

    # 2. Create blank base architecture model
    model = models.mobilenet_v2(pretrained=False)

    # 3. Adapt entry Conv layer: model.features[0][0] from 3 channels to 6 channels
    model.features[0][0] = nn.Conv2d(
        in_channels=in_channels, 
        out_channels=32, 
        kernel_size=(3, 3), 
        stride=(2, 2), 
        padding=(1, 1), 
        bias=False
    )

    if pretrained:
        state_dict_orig = model_orig.state_dict()
        state_dict_new = model.state_dict()

        for name, param in state_dict_orig.items():
            if name not in state_dict_new:
                continue
            if name == 'features.0.0.weight':
                # Splice original weights into the first half channels
                state_dict_new['features.0.0.weight'][:, :3, :, :] = param
                # Use Kaiming normal to initialize the newly added channels
                nn.init.kaiming_normal_(state_dict_new['features.0.0.weight'][:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            elif name in ['classifier.1.weight', 'classifier.1.bias']:
                continue  # Let classification mapping nodes bypass
            else:
                state_dict_new[name] = param

        model.load_state_dict(state_dict_new)
        print("Pretrained MobileNetV2 weights successfully loaded and adapted for 6 channels.")

    # 4. Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # FIX: Unfreeze the customized first conv layer to allow fusion channel adjustments
    for param in model.features[0][0].parameters():
        param.requires_grad = True

    # 5. Reconstruct classifier head matching output category density
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    return model


# ==============================================================================
# MOBILENETV2 LATE FUSION ARCHITECTURE (Feature-level Concat)
# ==============================================================================

class LateFusionMobileNet(nn.Module):
    """
    Dual-branch MobileNetV2 Late Fusion model.
    Extracts high-dimensional global descriptors up to the 1280-channel terminal,
    flattens via average pooling, and cascades representations into a linear network.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Instantiate independent feature extraction streams (.features blocks)
        self.mobilenet_branch1 = models.mobilenet_v2(pretrained=pretrained).features
        self.mobilenet_branch2 = models.mobilenet_v2(pretrained=pretrained).features

        # Freeze backbones parameters from gradient optimization updates
        for param in self.mobilenet_branch1.parameters():
            param.requires_grad = False
        for param in self.mobilenet_branch2.parameters():
            param.requires_grad = False

        # MobileNetV2 features layer yields 1280 channels.
        # Concatenation expands the combined vector slice: 1280 * 2 = 2560 dims
        fused_features_size = 1280 * 2

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(fused_features_size, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x1, x2):
        # Stream 1: Propagate -> Global Avg Pool -> Vector Flatten
        x1 = self.mobilenet_branch1(x1)
        x1 = nn.functional.adaptive_avg_pool2d(x1, (1, 1))
        x1 = torch.flatten(x1, 1)

        # Stream 2: Propagate -> Global Avg Pool -> Vector Flatten
        x2 = self.mobilenet_branch2(x2)
        x2 = nn.functional.adaptive_avg_pool2d(x2, (1, 1))
        x2 = torch.flatten(x2, 1)

        # Concatenate intermediate representations along feature boundaries
        fused_features = torch.cat((x1, x2), dim=1)
        
        # Dense mapping evaluation
        output = self.classifier(fused_features)
        return output