import os
import torch
import numpy as np
from sketch_gen import generate_sketch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def fetch_loader(ds_path, batch_size, image_size, sketch_path, gen_sketch=False):
    
    if gen_sketch:
        generate_sketch(ds_path, sketch_path)
        
    real_face_transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.ToTensor()
    ])
    sketch_face_transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.Normalize((0.5), (0.5))
        transforms.ToTensor()])
    
    real_face_dataset = datasets.ImageFolder(ds_path, transform = real_face_transform)
    sketch_face_dataset = datasets.ImageFolder(sketch_path, transform = sketch_face_transform)
    
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
