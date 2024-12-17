# trainer/dataset.py

import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

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