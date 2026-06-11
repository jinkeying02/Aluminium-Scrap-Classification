import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from utils.metrics import EarlyStopping, evaluate_and_print_metrics, plot_confusion_matrix

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
CONFIG = {
    'train_data_root': './data/train',
    'test_data_root': './data/test',
    'background_path': 'background_sample.png',
    
    # Modes: 'single' (no fusion) | 'early' (early fusion) | 'late' (late fusion)
    'experiment_mode': 'early', 
    
    # Valid only if experiment_mode is 'single': 'none' | 'blur' | 'removal'
    'baseline_bg_mode': 'none', 
    
    # Backbones: 'resnet18' | 'vgg16' | 'densenet121' | 'mobilenet_v2'
    'backbone': 'vgg16', 
    
    'batch_size': 16,
    'epochs': 50,
    'val_split_size': 0.2,
    'lr': 0.001,
    'weight_decay': 1e-3,
    'patience': 10
}

CONFIG['model_save_path'] = f"{CONFIG['backbone']}_{CONFIG['experiment_mode']}_trained.pth"

# ==============================================================================
# CORE BACKGROUND PROCESSORS
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
# EXPERIMENT DATASETS MATRIX
# ==============================================================================
class SingleStreamAluminiumDataset(Dataset):
    def __init__(self, root_dir, background_path, bg_mode='none', transform=None):
        self.transform, self.bg_mode = transform, bg_mode
        self.classes = ['Cast_Alu_Images', 'Wrought_Alu_Images']
        self.image_paths, self.labels = [], []
        self.back_img_bgr = cv2.imread(background_path)
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir): continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label_idx)
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.image_paths[idx])
        if self.bg_mode == 'removal': img_processed = extract_foreground_with_removal(img_bgr, self.back_img_bgr)
        elif self.bg_mode == 'blur': img_processed = extract_foreground_with_blur_bg(img_bgr, self.back_img_bgr)
        else: img_processed = img_bgr
        image = Image.fromarray(cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB))
        if self.transform: image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.long)

class EarlyFusionDataset(datasets.ImageFolder):
    def __init__(self, root, transform_raw=None, transform_processed=None):
        super().__init__(root, transform=None)
        self.transform_raw, self.transform_processed = transform_raw, transform_processed
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        fused_tensor = torch.cat((self.transform_raw(sample), self.transform_processed(sample)), dim=0)
        return fused_tensor, target

class LateFusionDataset(datasets.ImageFolder):
    def __init__(self, root, transform_branch1=None, transform_branch2=None):
        super().__init__(root, transform=None)
        self.transform_branch1, self.transform_branch2 = transform_branch1, transform_branch2
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        return self.transform_branch1(sample), self.transform_branch2(sample), target

# ==============================================================================
# MODEL ARCHITECTURES
# ==============================================================================
def get_single_stream_model(backbone_name, num_classes):
    if backbone_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif backbone_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif backbone_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    for param in model.parameters(): param.requires_grad = True 
    return model

def get_early_fusion_model(backbone_name, num_classes):
    if backbone_name == 'resnet18':
        model_orig = models.resnet18(pretrained=True)
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, num_classes)
        state_dict = model.state_dict()
        state_dict['conv1.weight'][:, :3, :, :] = model_orig.state_dict()['conv1.weight']
        model.load_state_dict(state_dict)
    elif backbone_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        orig_conv = model.features[0]
        model.features[0] = nn.Conv2d(6, orig_conv.out_channels, kernel_size=3, stride=1, padding=1)
        model.features[0].weight.data[:, :3, :, :] = orig_conv.weight.data
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif backbone_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.features.conv0 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif backbone_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        model.features[0][0] = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

class UnifiedLateFusionModel(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == 'resnet18':
            self.b1, self.b2 = models.resnet18(pretrained=True), models.resnet18(pretrained=True)
            self.b1.fc, self.b2.fc = nn.Identity(), nn.Identity()
            out_dim = 512 * 2
        elif backbone_name == 'vgg16':
            self.b1, self.b2 = models.vgg16(pretrained=True).features, models.vgg16(pretrained=True).features
            out_dim = (512 * 7 * 7) * 2
        elif backbone_name == 'densenet121':
            self.b1, self.b2 = models.densenet121(pretrained=True).features, models.densenet121(pretrained=True).features
            out_dim = 1024 * 2
        elif backbone_name == 'mobilenet_v2':
            self.b1, self.b2 = models.mobilenet_v2(pretrained=True).features, models.mobilenet_v2(pretrained=True).features
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
# TRAINING
# ==============================================================================
def train_experiment_engine(net, train_loader, val_loader, criterion, optimizer, device, is_late_fusion):
    net.to(device)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    early_stopping = EarlyStopping(patience=CONFIG['patience'], path=CONFIG['model_save_path'])

    for epoch in range(CONFIG['epochs']):
        net.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for data in train_loader:
            optimizer.zero_grad()
            if is_late_fusion:
                x1, x2, labels = data[0].to(device), data[1].to(device), data[2].to(device)
                outputs = net(x1, x2)
            else:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Validation phase
        net.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                if is_late_fusion:
                    x1, x2, labels = data[0].to(device), data[1].to(device), data[2].to(device)
                    outputs = net(x1, x2)
                else:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = net(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val

        print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        scheduler.step(epoch_val_loss)
        early_stopping(epoch_val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping triggered."); break

    net.load_state_dict(torch.load(CONFIG['model_save_path']))
    return history

def plot_history(history):
    epochs_range = np.arange(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.legend(); plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc')
    plt.legend(); plt.grid(); plt.show()

# ==============================================================================
# EXECUTION CONTROLLER ROUTINE
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing: backbone={CONFIG['backbone']} | mode={CONFIG['experiment_mode']} on {device}")

    # Standard feature transformation
    t_train_raw = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    t_val_test_raw = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    t_mog_train = transforms.Compose([MOGForegroundExtractor(CONFIG['background_path']), t_train_raw])
    t_mog_val = transforms.Compose([MOGForegroundExtractor(CONFIG['background_path']), t_val_test_raw])

    # Dynamic pipeline configuration routes
    mode = CONFIG['experiment_mode']
    if mode == 'single':
        full_dataset = SingleStreamAluminiumDataset(CONFIG['train_data_root'], CONFIG['background_path'], CONFIG['baseline_bg_mode'], t_train_raw)
        val_dataset_base = SingleStreamAluminiumDataset(CONFIG['train_data_root'], CONFIG['background_path'], CONFIG['baseline_bg_mode'], t_val_test_raw)
        model = get_single_stream_model(CONFIG['backbone'], num_classes=2)
    elif mode == 'early':
        full_dataset = EarlyFusionDataset(CONFIG['train_data_root'], t_train_raw, t_mog_train)
        val_dataset_base = EarlyFusionDataset(CONFIG['train_data_root'], t_val_test_raw, t_mog_val)
        model = get_early_fusion_model(CONFIG['backbone'], num_classes=2)
    elif mode == 'late':
        full_dataset = LateFusionDataset(CONFIG['train_data_root'], t_train_raw, t_mog_train)
        val_dataset_base = LateFusionDataset(CONFIG['train_data_root'], t_val_test_raw, t_mog_val)
        model = UnifiedLateFusionModel(CONFIG['backbone'], num_classes=2)

    # Stratified Dataset Splits
    targets = full_dataset.targets if hasattr(full_dataset, 'targets') else full_dataset.labels
    train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=CONFIG['val_split_size'], stratify=targets, random_state=42)
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(val_dataset_base, val_idx), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    # Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    history = train_experiment_engine(model, train_loader, val_loader, criterion, optimizer, device, is_late_fusion=(mode=='late'))
    plot_history(history)

    # ==========================================================================
    # EVALUATION
    # ==========================================================================
    print("\n--- Running Evaluation on Test Set ---")
    if mode == 'single': test_dataset = SingleStreamAluminiumDataset(CONFIG['test_data_root'], CONFIG['background_path'], CONFIG['baseline_bg_mode'], t_val_test_raw)
    elif mode == 'early': test_dataset = EarlyFusionDataset(CONFIG['test_data_root'], t_val_test_raw, t_mog_val)
    elif mode == 'late': test_dataset = LateFusionDataset(CONFIG['test_data_root'], t_val_test_raw, t_mog_val)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            if mode == 'late':
                outputs = model(data[0].to(device), data[1].to(device))
                labels = data[2]
            else:
                outputs = model(data[0].to(device))
                labels = data[1]
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.numpy())

    evaluate_and_print_metrics(y_true, y_pred, class_names=['Cast', 'Wrought'])
    plot_confusion_matrix(y_true, y_pred, classes=['Cast', 'Wrought'], save_path=f"{CONFIG['backbone']}_{mode}_confusion_matrix.png")
    
    print(f"\nAll experiments complete. Check weight at '{CONFIG['model_save_path']}' and image plot.")