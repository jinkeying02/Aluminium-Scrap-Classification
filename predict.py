import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# ==============================================================================
# PREDICTION CONFIGURATION
# ==============================================================================
CONFIG = {
    'single_image_path': r'E:\PythonProject\dis_cross\test\Wrought_Alu_Images\sample_test.png',
    'background_path': 'background_sample.png',
    
    # Must match the trained checkpoint: 'single' | 'early' | 'late'
    'experiment_mode': 'early', 
    
    # Valid only if experiment_mode is 'single': 'none' | 'blur' | 'removal'
    'baseline_bg_mode': 'none', 
    
    # Must match the trained backbone: 'resnet18' | 'vgg16' | 'densenet121' | 'mobilenet_v2'
    'backbone': 'vgg16',
    
    'num_classes': 2
}

# Auto-locate the corresponding checkpoint file
CONFIG['model_weights_path'] = f"{CONFIG['backbone']}_{CONFIG['experiment_mode']}_trained.pth"

# ==============================================================================
# PREPROCESSING
# ==============================================================================
def extract_foreground_with_removal(alu_img_bgr, back_img_bgr, threshold_value=10):
    if alu_img_bgr.shape != back_img_bgr.shape:
        back_img_bgr = cv2.resize(back_img_bgr, (alu_img_bgr.shape[1], alu_img_bgr.shape[0]))
    diff_img = cv2.subtract(alu_img_bgr, back_img_bgr)
    gray_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY) if len(diff_img.shape) == 3 else diff_img
    _, mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.bitwise_and(alu_img_bgr, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

def extract_foreground_with_blur_bg(alu_img_bgr, back_img_bgr, threshold_value=20):
    if alu_img_bgr.shape != back_img_bgr.shape:
        back_img_bgr = cv2.resize(back_img_bgr, (alu_img_bgr.shape[1], alu_img_bgr.shape[0]))
    diff_img = cv2.subtract(alu_img_bgr, back_img_bgr)
    gray_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY) if len(diff_img.shape) == 3 else diff_img
    _, initial_mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    processed_mask = cv2.dilate(cv2.erode(initial_mask, kernel, iterations=1), kernel, iterations=2)
    inverted_mask_3ch = cv2.cvtColor(cv2.bitwise_not(processed_mask), cv2.COLOR_GRAY2BGR)
    return np.where(inverted_mask_3ch == 255, alu_img_bgr, cv2.GaussianBlur(alu_img_bgr, (30, 30), 0))

class MOGForegroundExtractor:
    def __init__(self, background_image_path):
        self.background_image = cv2.imread(background_image_path)
    def __call__(self, img):
        original_img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        fgbg = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=150, detectShadows=False)
        if self.background_image is not None:
            fgbg.apply(self.background_image)
            mog_mask = fgbg.apply(original_img_bgr, learningRate=-1)
            kernel = np.ones((4, 4), np.uint8)
            mog_mask = cv2.dilate(cv2.erode(mog_mask, kernel, iterations=1), kernel, iterations=3)
            result = cv2.add(cv2.bitwise_and(original_img_bgr, original_img_bgr, mask=mog_mask),
                             cv2.bitwise_and(cv2.GaussianBlur(original_img_bgr, (21, 21), 0), cv2.GaussianBlur(original_img_bgr, (21, 21), 0), mask=cv2.bitwise_not(mog_mask)))
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return img

# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================
def get_single_stream_model(backbone_name, num_classes):
    if backbone_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == 'vgg16':
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif backbone_name == 'densenet121':
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif backbone_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

def get_early_fusion_model(backbone_name, num_classes):
    if backbone_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, num_classes)
    elif backbone_name == 'vgg16':
        model = models.vgg16(pretrained=False)
        model.features[0] = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif backbone_name == 'densenet121':
        model = models.densenet121(pretrained=False)
        model.features.conv0 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif backbone_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        model.features[0][0] = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

class UnifiedLateFusionModel(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == 'resnet18':
            self.b1, self.b2 = models.resnet18(pretrained=False), models.resnet18(pretrained=False)
            self.b1.fc, self.b2.fc = nn.Identity(), nn.Identity()
            out_dim = 512 * 2
        elif backbone_name == 'vgg16':
            self.b1, self.b2 = models.vgg16(pretrained=False).features, models.vgg16(pretrained=False).features
            out_dim = (512 * 7 * 7) * 2
        elif backbone_name == 'densenet121':
            self.b1, self.b2 = models.densenet121(pretrained=False).features, models.densenet121(pretrained=False).features
            out_dim = 1024 * 2
        elif backbone_name == 'mobilenet_v2':
            self.b1, self.b2 = models.mobilenet_v2(pretrained=False).features, models.mobilenet_v2(pretrained=False).features
            out_dim = 1280 * 2
            
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x1, x2):
        f1, f2 = self.b1(x1), self.b2(x2)
        if self.backbone_name in ['densenet121', 'mobilenet_v2']:
            f1 = torch.flatten(nn.functional.adaptive_avg_pool2d(f1, (1, 1)), 1)
            f2 = torch.flatten(nn.functional.adaptive_avg_pool2d(f2, (1, 1)), 1)
        else:
            f1, f2 = torch.flatten(f1, 1), torch.flatten(f2, 1)
        return self.classifier(torch.cat((f1, f2), dim=1))

# ==============================================================================
# IMAGE INFERENCE
# ==============================================================================
def inference_single_image():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mode = CONFIG['experiment_mode']
    
    # Standard feature transformation
    t_base = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Re-instantiate the precise structural target model
    if mode == 'single':   model = get_single_stream_model(CONFIG['backbone'], CONFIG['num_classes'])
    elif mode == 'early': model = get_early_fusion_model(CONFIG['backbone'], CONFIG['num_classes'])
    elif mode == 'late':  model = UnifiedLateFusionModel(CONFIG['backbone'], CONFIG['num_classes'])

    # Load parameters mapping dictionary onto skeleton structure
    model.load_state_dict(torch.load(CONFIG['model_weights_path'], map_location=device))
    model.to(device)
    model.eval()

    # Process image loading routes
    img_bgr = cv2.imread(CONFIG['single_image_path'])
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to load target raw image from: {CONFIG['single_image_path']}")
        
    back_img_bgr = cv2.imread(CONFIG['background_path'])

    if mode == 'single':
        # Apply specific digital image preprocessing baseline metrics
        if CONFIG['baseline_bg_mode'] == 'removal':
            img_processed = extract_foreground_with_removal(img_bgr, back_img_bgr)
        elif CONFIG['baseline_bg_mode'] == 'blur':
            img_processed = extract_foreground_with_blur_bg(img_bgr, back_img_bgr)
        else:
            img_processed = img_bgr
            
        pil_img = Image.fromarray(cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB))
        input_tensor = t_base(pil_img).unsqueeze(0).to(device)

    else:
        # Multi-Stream processing (Raw Image Stream + MOG2 Background Stream)
        mog_extractor = MOGForegroundExtractor(CONFIG['background_path'])
        
        pil_raw = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pil_mog = mog_extractor(pil_raw)
        
        tensor_raw = t_base(pil_raw)
        tensor_mog = t_base(pil_mog)

        if mode == 'early':
            # Channel-level stack: Yields single [1, 6, 224, 224] tensor batch
            input_tensor = torch.cat((tensor_raw, tensor_mog), dim=0).unsqueeze(0).to(device)
        elif mode == 'late':
            # Deep feature level stack: Yields two independent [1, 3, 224, 224] tensor vectors
            input_tensor_x1 = tensor_raw.unsqueeze(0).to(device)
            input_tensor_x2 = tensor_mog.unsqueeze(0).to(device)

    # Evaluation
    classes = ['Cast_Aluminium', 'Wrought_Aluminium']

    with torch.no_grad():
        if mode == 'late':
            outputs = model(input_tensor_x1, input_tensor_x2)
        else:
            outputs = model(input_tensor)
            
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    # Output metrics
    print("\n" + "="*40)
    print("METALLURGIC SCRAP INFERENCE OUTPUT")
    print("="*40)
    print(f"Target file : {CONFIG['single_image_path']}")
    print(f"Evaluated via: {CONFIG['backbone']} ({mode} fusion platform)")
    print(f"Class Verdict: {classes[predicted_idx.item()]}")
    print(f"Confidence   : {confidence.item() * 100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    inference_single_image()