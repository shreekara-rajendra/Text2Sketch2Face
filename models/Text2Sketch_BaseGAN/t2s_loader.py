import os
import pandas as pd
from skimage import io
import torch
import numpy as np
from sketch_gen import generate_sketch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CreateDataLoader(Dataset):
    def __init__(self, csv_file, root_dir_real, root_dir_sketch, transform=None):
        self.data = pd.read_csv(csv_file, sep='\t')
        self.root_dir_real = root_dir_real
        self.root_dir_sketch = root_dir_sketch
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir_sketch) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.data.iloc[idx]
        image_filename = row[0].split(',')[0] 
        sketch_img_name = os.path.join(self.root_dir_sketch, image_filename)
        sketch_image = io.imread(sketch_img_name)
        attributes = row[0].split(',')[1:]
        if self.transform:
            sketch_image = self.transform(sketch_image)
        item = {'img': sketch_image, 'att': torch.Tensor([float(attr) for attr in attributes])}
        return item
    
    
    

def fetch_loader(ds_path, csv, batch_size, image_size, sketch_path, gen_sketch=False):
    
    if gen_sketch:
        generate_sketch(ds_path, sketch_path)
        
   
    sketch_face_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize(np.array([0.5]), np.array([0.5]))
    ])
    
    sketch_face_dataset = CreateDataLoader(csv, ds_path,sketch_path, transform=sketch_face_transform)

    sketch_face_dataloader = DataLoader(dataset = sketch_face_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=8,
                                      drop_last=True)
    
    
    return sketch_face_dataloader
