import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

    
def feature_encoding(sketch):
    '''
    # Convert 1 channel to 3
    sketch = torch.randn(size=[1, 1, 64, 64])
    '''
    sketch = torch.cat([sketch] * 3, dim=1)  # Concatenate along the channel dimension
    print(sketch.shape)

    vgg16 = models.vgg16(pretrained=True)
    # Remove fully connected layers
    feature_encoding_model = nn.Sequential(*list(vgg16.features.children())[:4])  # Extract layers up to conv1_2
    # Set the model to evaluation mode
    feature_encoding_model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        # Remove transforms.ToTensor() since input is already a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_image = preprocess(sketch)  # No need to unsqueeze(0) as the batch dimension is already there
    print(input_image.shape)

    # Obtain feature representation
    with torch.no_grad():
        features = feature_encoding_model(input_image)

    # Now 'features' contains the output from the conv1_2 layer
    print(features.shape)

def perceptual_loss(real,fake):
    loss_fn = nn.L1Loss()
    loss = loss_fn(fake,real)
    return loss

'''
testing
'''

t1 = torch.randn(size = [1,1,64,64])
t2 = torch.randn(size = [1,1,64,64])
print(perceptual_loss(t1,t2))