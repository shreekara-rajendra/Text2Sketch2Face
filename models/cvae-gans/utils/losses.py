import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import math

## perceptual loss for G2
class perceptual_loss(nn.Module):
    def __init__(self):
        super(perceptual_loss,self).__init__()
        self.loss_fn = nn.L1Loss()
    
    @staticmethod
    def feature_encoding(sketch):
        sketch = torch.cat([sketch] * 3, dim=1)  # Concatenate along the channel dimension

        vgg16 = models.vgg16(pretrained=True)
        # Remove fully connected layers
        feature_encoding_model = nn.Sequential(*list(vgg16.features.children())[:4])  # Extract layers up to conv1_2
        # Set the model to evaluation mode
        feature_encoding_model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_image = preprocess(sketch)  # No need to unsqueeze(0) as the batch dimension is already there

        # Obtain feature representation
        with torch.no_grad():
            features = feature_encoding_model(input_image)

        # Now 'features' contains the output from the conv1_2 layer
        return features
        
    def __call__(self,real,fake):
        real = self.feature_encoding(real)
        fake = self.feature_encoding(fake)
        loss = self.loss_fn(fake,real)
        return loss

## patch_gan loss if input is real
class patch_real(nn.Module):
    def __init__(self):
        super(patch_real,self).__init__()
        self.loss = nn.BCELoss()
        
    def __call__(self,sketch):
        reals = torch.ones(sketch.size(),requires_grad = False)
        return self.loss(sketch,reals)
    
## patch_gan loss if input is fake
class patch_fake(nn.Module):
    def __init__(self):
        super(patch_fake,self).__init__()
        self.loss = nn.BCELoss()
    
    def __call__(self,sketch):
        fakes = torch.zeros(sketch.size(),requires_grad = False)
        return self.loss(sketch,fakes)

## reconstruction loss for G2
class reconstruction_loss(nn.Module):
    def __init__(self):
        super(reconstruction_loss,self).__init__()
        self.loss = nn.L1Loss()
    
    def __call__(self,target,generated):
        return self.loss(target,generated)

## KL-divergence is 1/2 * (mean^2 + sigma^2 - 1 - logvar)
def KLDivergence(mean, logvar):
    sigma_squared = torch.exp(logvar)
    mean_squared = mean.pow(2)
    expr = sigma_squared + mean_squared - logvar -1
    kld = 0.5 * expr
    loss = torch.mean(kld)
    return loss

## maximum likelihood (gausian similarity) => 1/2 * (log(2*pie*variance)+((y-mean)**2)/variance
def GausianSimilarity(pred, target):
    mean = pred[0]
    logvar = pred[1]
    term1 = logvar + math.log(2*math.pi)
    ## target is an image. trying to find how best the predicted distribution fits the image
    term2 = ((target - mean)**2)/(torch.exp(logvar))
    loss = 0.5 * (term1 + term2)
    return loss

pp = perceptual_loss()
t1 = torch.randn(size = [1,1,64,64])
t2 = torch.randn(size = [1,1,64,64])
print(pp(t1,t2))

