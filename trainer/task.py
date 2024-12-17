# trainer/task.py

import os
from google.cloud import storage
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from trainer.yolo import YOLO  
from trainer.train import YOLOLoss
from dataset import COCODataset 
from trainer.train import train_one_epoch, validate

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def train_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create local directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(f"{args.data_dir}/train2017", exist_ok=True)
    os.makedirs(f"{args.data_dir}/val2017", exist_ok=True)
    os.makedirs(f"{args.data_dir}/annotations", exist_ok=True)
    
    # Download data from GCS
    print("Downloading dataset from GCS...")
    download_from_gcs(
        args.bucket_name,
        f"{args.gcs_data_path}/train2017",
        f"{args.data_dir}/train2017"
    )
    download_from_gcs(
        args.bucket_name,
        f"{args.gcs_data_path}/val2017",
        f"{args.data_dir}/val2017"
    )
    download_from_gcs(
        args.bucket_name,
        f"{args.gcs_data_path}/annotations/instances_train2017.json",
        f"{args.data_dir}/annotations/instances_train2017.json"
    )
    download_from_gcs(
        args.bucket_name,
        f"{args.gcs_data_path}/annotations/instances_val2017.json",
        f"{args.data_dir}/annotations/instances_val2017.json"
    )
    
    # Initialize model, dataset, and training components
    model = YOLO(num_classes=80).to(device)
    
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = COCODataset(
        img_dir=f"{args.data_dir}/train2017",
        annotation_file=f"{args.data_dir}/annotations/instances_train2017.json",
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create validation dataset and loader
    val_dataset = COCODataset(
        img_dir=f"{args.data_dir}/val2017",
        annotation_file=f"{args.data_dir}/annotations/instances_val2017.json",
        transform=transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    criterion = YOLOLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[80, 110],
        gamma=0.1
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Epoch {epoch + 1}/{args.num_epochs}')
        print(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = f"checkpoints/best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            
            # Upload best model to GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(args.bucket_name)
            blob = bucket.blob(f"{args.gcs_output_path}/checkpoints/best_model.pth")
            blob.upload_from_filename(checkpoint_path)
        
        # Save regular checkpoint at intervals
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            
            # Upload to GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(args.bucket_name)
            blob = bucket.blob(f"{args.gcs_output_path}/checkpoints/checkpoint_epoch_{epoch+1}.pth")
            blob.upload_from_filename(checkpoint_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', type=str, default='asuran-coco-data')
    parser.add_argument('--gcs-data-path', type=str, default='Dataset')
    parser.add_argument('--gcs-output-path', type=str, default='training_outputs')
    parser.add_argument('--data-dir', type=str, default='/tmp/data')
    parser.add_argument('--num-epochs', type=int, default=135)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-interval', type=int, default=5)
    
    args = parser.parse_args()
    train_model(args)