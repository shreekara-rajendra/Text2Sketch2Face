import os
import torch
import numpy as np
from sketch_gen import generate_sketch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

def fetch_loader(ds_path, batch_size, image_size, sketch_path, gen_sketch=False):
    
    if gen_sketch:
        generate_sketch(ds_path, sketch_path)
        
    real_face_transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
    ])
    sketch_face_transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize(np.array([0.5]), np.array([0.5]))
    ])
    
    real_face_dataset = CustomDataset(data_dir=ds_path, transform=real_face_transform)
    sketch_face_dataset = CustomDataset(data_dir=sketch_path, transform=sketch_face_transform)
    
    real_face_dataloader = DataLoader(dataset = real_face_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=8,
                                      drop_last=True)
    
    sketch_face_dataloader = DataLoader(dataset = sketch_face_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=8,
                                      drop_last=True)
    
    
    return real_face_dataloader, sketch_face_dataloader
