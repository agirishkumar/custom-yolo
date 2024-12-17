# trainer/train.py

import torch
import torch.nn as nn
from tqdm import tqdm

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

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} Training')

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        if batch_idx % 100 == 0:
            print(f'\nBatch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'val_loss': loss.item()})
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss