# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from yolo import YOLO

class COCODataset(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            coco = json.load(f)
            
        # Get category information
        self.categories = coco['categories']
        self.cat_ids = [cat['id'] for cat in self.categories]
        self.max_cat_id = max(self.cat_ids)
        self.cat_id_to_continuous = {cat_id: idx for idx, cat_id in enumerate(sorted(self.cat_ids))}
        
        print(f"Number of categories: {len(self.categories)}")
        print(f"Category ID range: {min(self.cat_ids)} to {max(self.cat_ids)}")
        
        # Filter out images that don't exist in the directory
        self.images = []
        for img in coco['images']:
            if os.path.exists(os.path.join(img_dir, img['file_name'])):
                self.images.append(img)
        
        print(f"Found {len(self.images)} valid images out of {len(coco['images'])} annotations")
        
        self.annotations = coco['annotations']
        
        # Create image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        # Create image_id to image mapping
        self.id_to_img = {img['id']: img for img in self.images}
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        # Create target tensor (7x7x(5*2+80))
        target = torch.zeros((7, 7, 90))  # 2 boxes * 5 + 80 classes
        
        # Convert annotations to YOLO format
        orig_w, orig_h = img_info['width'], img_info['height']
        
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            cat_id = self.cat_id_to_continuous[ann['category_id']]  # Convert to continuous 0-based index
            
            # Convert bbox to center format and normalize
            x_center = (bbox[0] + bbox[2]/2) / orig_w
            y_center = (bbox[1] + bbox[3]/2) / orig_h
            width = bbox[2] / orig_w
            height = bbox[3] / orig_h
            
            # Get grid cell location
            grid_x = int(x_center * 7)
            grid_y = int(y_center * 7)
            
            # Adjust for edge case
            grid_x = min(6, grid_x)
            grid_y = min(6, grid_y)
            
            # If no object is assigned to this cell yet
            if target[grid_y, grid_x, 4] == 0:
                # Set box coordinates
                target[grid_y, grid_x, 0:4] = torch.tensor([x_center, y_center, width, height])
                target[grid_y, grid_x, 4] = 1  # confidence
                target[grid_y, grid_x, 10 + cat_id] = 1  # class probability
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, target

class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def compute_iou(self, box1, box2):
        """Compute IOU between two boxes"""
        # Convert to x1, y1, x2, y2 format
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2
        
        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
        
        # Intersection area
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, 0) * torch.clamp(inter_y2 - inter_y1, 0)
        
        # Union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def forward(self, predictions, targets):
        """
        predictions: (batch_size, 7, 7, 90)
        targets: (batch_size, 7, 7, 90)
        """
        batch_size = predictions.size(0)
        
        # Reshape predictions
        pred_boxes = predictions[..., :10].reshape(batch_size, 7, 7, 2, 5)  # (batch_size, 7, 7, 2, 5)
        pred_classes = predictions[..., 10:]  # (batch_size, 7, 7, 80)
        
        # Get target components
        target_boxes = targets[..., :5]  # (batch_size, 7, 7, 5)
        target_classes = targets[..., 10:]  # (batch_size, 7, 7, 80)
        
        # Mask for cells that contain objects (batch_size, 7, 7, 1)
        obj_mask = target_boxes[..., 4].unsqueeze(-1)
        noobj_mask = 1 - obj_mask
        
        # Split prediction boxes
        pred_box1 = pred_boxes[..., 0, :]  # (batch_size, 7, 7, 5)
        pred_box2 = pred_boxes[..., 1, :]  # (batch_size, 7, 7, 5)
        
        # Compute IOUs for both predicted boxes
        iou1 = self.compute_iou(pred_box1[..., :4], target_boxes[..., :4])  # (batch_size, 7, 7)
        iou2 = self.compute_iou(pred_box2[..., :4], target_boxes[..., :4])  # (batch_size, 7, 7)
        
        # Create responsible box mask
        responsible_mask = (iou1 > iou2).unsqueeze(-1)  # (batch_size, 7, 7, 1)
        
        # Get responsible box predictions
        responsible_pred = torch.where(
            responsible_mask,
            pred_box1,
            pred_box2
        )
        
        # Coordinate loss
        coord_mask = obj_mask  # (batch_size, 7, 7, 1)
        xy_loss = coord_mask * torch.sum((responsible_pred[..., :2] - target_boxes[..., :2])**2, dim=-1, keepdim=True)
        wh_loss = coord_mask * torch.sum(
            (torch.sqrt(responsible_pred[..., 2:4] + 1e-6) - torch.sqrt(target_boxes[..., 2:4] + 1e-6))**2,
            dim=-1, keepdim=True
        )
        coord_loss = self.lambda_coord * (xy_loss + wh_loss).sum()
        
        # Confidence loss
        responsible_conf = torch.where(
            responsible_mask,
            pred_box1[..., 4:5],
            pred_box2[..., 4:5]
        )
        
        obj_conf_loss = obj_mask * (responsible_conf - 1)**2
        noobj_conf_loss = noobj_mask * (responsible_conf)**2
        
        conf_loss = obj_conf_loss.sum() + self.lambda_noobj * noobj_conf_loss.sum()
        
        # Class loss
        class_loss = obj_mask * torch.sum((pred_classes - target_classes)**2, dim=-1, keepdim=True)
        class_loss = class_loss.sum()
        
        # Total loss
        total_loss = (coord_loss + conf_loss + class_loss) / batch_size
        
        return total_loss

    
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        print(f"Output Shape: {output.shape}, Target Shape: {target.shape}")  # Debugging line
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Print batch statistics
        if batch_idx % 100 == 0:
            print(f'\nBatch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def custom_collate(batch):
    # Filter out None values
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if len(batch) == 0:
        raise RuntimeError("Empty batch")
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 135
    learning_rate = 0.001
    
    # Data paths
    train_img_dir = "Dataset/train2017"
    train_ann_file = "Dataset/annotations/instances_train2017.json"
    
    # Model
    model = YOLO(num_classes=80).to(device)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and DataLoader
    train_dataset = COCODataset(train_img_dir, train_ann_file, transform=transform)
    train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4,
    collate_fn=custom_collate
)
    
    # Loss and optimizer
    criterion = YOLOLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 110], gamma=0.1)
    
    # Training loop
    for epoch in range(num_epochs):
        avg_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
            
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    main()