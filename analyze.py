import json
import os
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

class COCOAnalyzer:
    def __init__(self, annotation_path, image_dir):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        with open(annotation_path, 'r') as f:
            self.coco_data = json.load(f)
            
    def get_basic_stats(self):
        """Get basic dataset statistics"""
        stats = {
            'num_images': len(self.coco_data['images']),
            'num_annotations': len(self.coco_data['annotations']),
            'num_categories': len(self.coco_data['categories']),
        }
        
        # Calculate average annotations per image
        stats['avg_annotations_per_image'] = stats['num_annotations'] / stats['num_images']
        
        return stats
    
    def analyze_categories(self):
        """Analyze category distribution"""
        # Create category id to name mapping
        cat_mapping = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Count instances per category
        category_counts = Counter(ann['category_id'] for ann in self.coco_data['annotations'])
        
        # Convert to readable format
        category_stats = {cat_mapping[cat_id]: count 
                         for cat_id, count in category_counts.items()}
        
        return category_stats
    
    def analyze_image_sizes(self):
        """Analyze image dimensions"""
        widths = []
        heights = []
        aspects = []
        
        for img in self.coco_data['images']:
            widths.append(img['width'])
            heights.append(img['height'])
            aspects.append(img['width'] / img['height'])
            
        size_stats = {
            'width': {'mean': np.mean(widths), 'std': np.std(widths),
                     'min': min(widths), 'max': max(widths)},
            'height': {'mean': np.mean(heights), 'std': np.std(heights),
                      'min': min(heights), 'max': max(heights)},
            'aspect_ratio': {'mean': np.mean(aspects), 'std': np.std(aspects),
                           'min': min(aspects), 'max': max(aspects)}
        }
        
        return size_stats
    
    def analyze_bbox_sizes(self):
        """Analyze bounding box dimensions"""
        widths = []
        heights = []
        areas = []
        
        for ann in self.coco_data['annotations']:
            bbox = ann['bbox']  # [x, y, width, height]
            widths.append(bbox[2])
            heights.append(bbox[3])
            areas.append(bbox[2] * bbox[3])
            
        bbox_stats = {
            'width': {'mean': np.mean(widths), 'std': np.std(widths),
                     'min': min(widths), 'max': max(widths)},
            'height': {'mean': np.mean(heights), 'std': np.std(heights),
                      'min': min(heights), 'max': max(heights)},
            'area': {'mean': np.mean(areas), 'std': np.std(areas),
                    'min': min(areas), 'max': max(areas)}
        }
        
        return bbox_stats
    
    def analyze_objects_per_image(self):
        """Analyze number of objects per image"""
        img_ann_count = Counter(ann['image_id'] for ann in self.coco_data['annotations'])
        counts = list(img_ann_count.values())
        
        objects_stats = {
            'mean': np.mean(counts),
            'std': np.std(counts),
            'min': min(counts),
            'max': max(counts),
            'distribution': Counter(counts)
        }
        
        return objects_stats

def main():
    # Paths
    train_annot_path = "Dataset/annotations/instances_train2017.json"
    train_img_dir = "Dataset/train2017"
    val_annot_path = "Dataset/annotations/instances_val2017.json"
    val_img_dir = "Dataset/val2017"
    
    # Analyze training set
    print("\nAnalyzing Training Set...")
    train_analyzer = COCOAnalyzer(train_annot_path, train_img_dir)
    
    # Basic stats
    train_stats = train_analyzer.get_basic_stats()
    print("\nBasic Statistics (Training):")
    for k, v in train_stats.items():
        print(f"{k}: {v}")
    
    # Category distribution
    category_stats = train_analyzer.analyze_categories()
    print("\nTop 10 Categories by Instance Count:")
    for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{cat}: {count}")
    
    # Image size analysis
    size_stats = train_analyzer.analyze_image_sizes()
    print("\nImage Size Statistics:")
    print(f"Average dimensions: {size_stats['width']['mean']:.1f} x {size_stats['height']['mean']:.1f}")
    print(f"Aspect ratio range: {size_stats['aspect_ratio']['min']:.2f} to {size_stats['aspect_ratio']['max']:.2f}")
    
    # Bounding box analysis
    bbox_stats = train_analyzer.analyze_bbox_sizes()
    print("\nBounding Box Statistics:")
    print(f"Average size: {bbox_stats['width']['mean']:.1f} x {bbox_stats['height']['mean']:.1f}")
    print(f"Average area: {bbox_stats['area']['mean']:.1f}")
    
    # Objects per image analysis
    objects_stats = train_analyzer.analyze_objects_per_image()
    print("\nObjects per Image:")
    print(f"Average: {objects_stats['mean']:.1f}")
    print(f"Range: {objects_stats['min']} to {objects_stats['max']}")

if __name__ == "__main__":
    main()