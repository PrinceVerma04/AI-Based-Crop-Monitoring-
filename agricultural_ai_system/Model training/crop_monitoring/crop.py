import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from PIL import Image

class CropDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.transform = transform
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def setup_model(num_classes):
    model = torchvision.models.efficientnet_b4(pretrained=True)
    
    # Freeze all layers except final classification layer
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
    
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(dataloaders[phase], desc=f'Epoch {epoch} {phase}'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Epoch {epoch}/{num_epochs-1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return model

def save_model_artifacts(model, class_names, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(output_dir, "crop_classifier.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save class names
    class_file = os.path.join(output_dir, "class_names.txt")
    with open(class_file, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"Model artifacts saved to {output_dir}")

def main():
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Prepare dataset
    dataset_path = "/workspaces/AgriTech-Hackathon/agricultural_ai_system/Model training/datasets/crop/Agricultural-crops"
    dataset = CropDataset(dataset_path, transform=None)
    
    # Split dataset
    train_idx, val_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        stratify=[img[1] for img in dataset.images]
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
    }

    # Initialize model
    model = setup_model(num_classes=len(dataset.classes))
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5)
    
    # Save model artifacts
    save_model_artifacts(model, dataset.classes)

if __name__ == "__main__":
    main()