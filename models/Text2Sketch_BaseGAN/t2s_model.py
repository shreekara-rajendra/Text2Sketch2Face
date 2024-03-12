import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, output_padding=pad, bias=not use_bn))
    else:
        module.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = conv_block(channels, channels, k_size=3, stride=1, pad=1, use_bn=True)
        self.conv2 = conv_block(channels, channels, k_size=3, stride=1, pad=1, use_bn=True)

    def __call__(self, x):
        x = F.relu(self.conv1(x))
        return x + self.conv2(x)


class Discriminator2(nn.Module):
    def __init__(self, channels=3, conv_dim=64):
        super(Discriminator2, self).__init__()
        self.conv1 = conv_block(channels, conv_dim, use_bn=False)
        self.conv2 = conv_block(conv_dim, conv_dim * 2)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4)
        self.conv4 = conv_block(conv_dim * 4, 1, k_size=3, stride=1, pad=1, use_bn=False)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        alpha = 0.2
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = F.leaky_relu(self.conv3(x), alpha)
        x = self.conv4(x)
        x = x.reshape([x.shape[0], -1]).mean(1)
        return x



class Discriminator(nn.Module):
    def __init__(self, channels=3, conv_dim=64, num_classes=40):
        super(Discriminator, self).__init__()
        self.conv1 = conv_block(channels, conv_dim, use_bn=False)
        self.conv2 = conv_block(conv_dim, conv_dim * 2)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(conv_dim * 4 * (64 // 8) * (64 // 8), 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Activation
        self.sigmoid = nn.Sigmoid()

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        alpha = 0.2
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = F.leaky_relu(self.conv3(x), alpha)

        # Flatten the feature map
        x = self.flatten(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Sigmoid activation for multi-label classification
        x = self.sigmoid(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiheadAttention, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        
        self.query_linear = nn.Linear(input_size, input_size)
        self.key_linear = nn.Linear(input_size, input_size)
        self.value_linear = nn.Linear(input_size, input_size)
        
        self.output_linear = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)
        Q = Q.view(Q.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(K.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), -1, self.input_size)
        output = self.output_linear(attn_output)
        return output



class Generator(nn.Module):
    def __init__(self, input_size=40, out_channels=1, conv_dim=32):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        # Dense layer to project latent space
        input_size = 40
        num_heads = 4
        self.multihead_attention = MultiheadAttention(input_size, num_heads)
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 8 * 8 * conv_dim)  # Mapping to 16*16*conv_dim

        self.res1 = ResBlock(conv_dim)
        self.res2 = ResBlock(conv_dim)
        self.res3 = ResBlock(conv_dim)
        self.tconv4 = conv_block(conv_dim, conv_dim * 2, k_size=3, stride=2, pad=1, use_bn=True, transpose=True)
        self.tconv5 = conv_block(conv_dim * 2, conv_dim * 4, k_size=3, stride=2, pad=1, use_bn=True, transpose=True)
        self.tconv6 = conv_block(conv_dim * 4, conv_dim * 8, k_size=3, stride=2, pad=1, use_bn=True, transpose=True)
        self.conv7 = conv_block(conv_dim * 8, out_channels, k_size=3, stride=1, pad=1, use_bn=False)

    def forward(self, x):
        x = self.multihead_attention(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, self.conv_dim, 8, 8)  # Reshape to match the input of the next layer
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = F.relu(self.tconv4(x))
        x = F.relu(self.tconv5(x))
        x = F.relu(self.tconv6(x))
        x = torch.tanh(self.conv7(x))
        return x

# # Example usage
# # Initialize the generator
# generator = Generator()

# # Generate random input tensor
# batch_size = 10
# input_tensor = torch.randn(batch_size, 40)

# # Generate images
# generated_images = generator(input_tensor)

# # Check the shape of the generated images
# print(generated_images.shape)  # Output: torch.Size([10, 1, 64, 64])
