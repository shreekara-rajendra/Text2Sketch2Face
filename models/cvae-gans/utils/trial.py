import torch
import torch.nn.functional as F

t1 = torch.randn(size=[1, 2, 1,1])  # Adding one more spatial dimension
print("Original tensor:")
print(t1)



t2 = F.interpolate(t1, size=[2, 2], mode='nearest')
print("\nInterpolated tensor:")
print(t2.shape)  