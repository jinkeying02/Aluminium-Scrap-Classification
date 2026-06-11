import torch
import torch.nn as nn
import torchvision.models as models

# ==============================================================================
# DENSENET121 EARLY FUSION ARCHITECTURE (Channel-level Concat)
# ==============================================================================

def densenet121_early_fusion(num_classes=2, in_channels=6, pretrained=True):
    """
    Factory function adapting DenseNet-121 for Early Fusion.
    Modifies the entry layer 'features.conv0' to ingest 6-channel multi-stream tensors
    and injects pretrained 3-channel weights into the first half.
    """
    # 1. Load standard model for pretrained weights sourcing
    model_orig = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)

    # 2. Create the blank target model skeleton
    model = models.densenet121(weights=None)

    # 3. Adapt the entry convolution layer (features.conv0)
    model.features.conv0 = nn.Conv2d(
        in_channels=in_channels, 
        out_channels=64, 
        kernel_size=(7, 7), 
        stride=(2, 2), 
        padding=(3, 3), 
        bias=False
    )

    if pretrained:
        state_dict_orig = model_orig.state_dict()
        state_dict_new = model.state_dict()

        for name, param in state_dict_orig.items():
            if name not in state_dict_new:
                continue
            if name == 'features.conv0.weight':
                # Map ImageNet weights into channels 0, 1, 2
                state_dict_new['features.conv0.weight'][:, :3, :, :] = param
                # Initialize remaining channels using Kaiming normal distribution
                nn.init.kaiming_normal_(state_dict_new['features.conv0.weight'][:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            elif name in ['classifier.weight', 'classifier.bias']:
                continue  # Let classification nodes realign to local dimensions
            else:
                state_dict_new[name] = param

        model.load_state_dict(state_dict_new)
        print("Pretrained DenseNet-121 weights adapted for 6-channel Early Fusion entry.")

    # 4. Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Unfreeze the modified entry conv0 layer to update custom fusion channel kernels
    for param in model.features.conv0.parameters():
        param.requires_grad = True

    # 5. Replace classifier head mapping the target category outputs
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    return model


# ==============================================================================
# DENSENET121 LATE FUSION ARCHITECTURE (Feature-level Concat)
# ==============================================================================

class LateFusionDenseNet(nn.Module):
    """
    Dual-branch DenseNet-121 Late Fusion model.
    Extracts deep representation cubes up to the terminal global block,
    flattens via adaptive pooling, and chains them prior to joint logit scoring.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None

        # Build independent feature extraction streams (.features containers only)
        self.densenet_branch1 = models.densenet121(weights=weights).features
        self.densenet_branch2 = models.densenet121(weights=weights).features

        # Freeze parameter optimization across both trunks
        for param in self.densenet_branch1.parameters():
            param.requires_grad = False
        for param in self.densenet_branch2.parameters():
            param.requires_grad = False

        # DenseNet121 features map natively to 1024 dimensions.
        # Concatenation doubles the combined vector scope: 1024 * 2 = 2048 dims
        fused_features_size = 1024 * 2

        self.classifier = nn.Sequential(
            nn.Linear(fused_features_size, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x1, x2):
        # Stream 1: Forward propagate -> Adaptive Global Avg Pool -> Flatten Vector
        x1 = self.densenet_branch1(x1)
        x1 = nn.functional.adaptive_avg_pool2d(x1, (1, 1))
        x1 = torch.flatten(x1, 1)

        # Stream 2: Forward propagate -> Adaptive Global Avg Pool -> Flatten Vector
        x2 = self.densenet_branch2(x2)
        x2 = nn.functional.adaptive_avg_pool2d(x2, (1, 1))
        x2 = torch.flatten(x2, 1)

        # Splice dual representations channel-wise
        fused_features = torch.cat((x1, x2), dim=1)
        
        return self.classifier(fused_features)