import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import random
import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum
from .blocks import VA


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

RESNET_CONFIGS = {'18': [[2, 2, 2, 2], PreActBlock],
                  '28': [[3, 4, 6, 3], PreActBlock],
                  '34': [[3, 4, 6, 3], PreActBlock],
                  '50': [[3, 4, 6, 3], PreActBottleneck],
                  '101': [[3, 4, 23, 3], PreActBottleneck]
                  }

def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False

class VASet(nn.Module):
    def __init__(self, channels, T):
        super().__init__()
        self.va = VA(channels)
        self.conv = nn.Conv2d(channels*T, channels*T, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(channels*T)
        self.t = T
        self.activation = nn.ReLU()

    def forward(self, x):
        x = rearrange(x, "b (t c) h w -> b c t h w", t=self.t)
        attention = self.va(x)
        attention = rearrange(attention, " b c t h w -> b (t c) h w")
        attention = self.activation(self.bn(self.conv(attention)))
        return attention

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn
    
class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut 
        return x

class ResNet(nn.Module):
    def __init__(self, dim, resnet_type='18'):
        super(ResNet, self).__init__()

        layers, block = RESNET_CONFIGS[resnet_type]
        self.activation = nn.ReLU()
        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(1, int(dim/4), kernel_size=3, stride=2, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(int(dim/4))

        self.conv11 = nn.Conv2d(1, int(dim/4), kernel_size=3, stride=2, padding=(1, 1), bias=False)
        self.bn11 = nn.BatchNorm2d(int(dim/4))

        self.in_planes = int(dim/4)
        self.layer1 = self._make_layer(block, int(dim/2), layers[0], stride=2)
        self.layer2 = self._make_layer(block, dim, layers[0], stride=2)
        self.layer3 = self._make_layer(block, dim*2, layers[1], stride=2)
        self.layer4 = self._make_layer(block, dim*4, layers[1], stride=2)

        self.in_planes = int(dim/4)
        self.layer11 = self._make_layer(block, int(dim/2), layers[0], stride=2)
        self.layer22 = self._make_layer(block, dim, layers[0], stride=2)

        self.van1 = Attention(1)
        self.van2 = Attention(1)

        self.conv0 = nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.bn0 = nn.BatchNorm2d(dim)

        self.conv2 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(dim*2)

        self.conv3 = nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(dim*4)

        # self.initialize_params()

        self.va1 = VASet(dim*2, 5)
        self.va2 = VASet(dim*4, 5)
        self.va3 = VASet(dim*4, 5)


    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = rearrange(x, "b c t h w -> (b t) c h w ", t = 5)

        x2 = x.clone()
        x2[x2 < 180/255] = 0

        x = self.van1(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.van1(x)

        x2 = self.van2(x2)
        x2 = self.activation(self.bn11(self.conv11(x2)))
        x2 = self.layer11(x2)
        x2 = self.layer22(x2)
        # x2 = self.van2(x2)

        x = torch.cat([x,x2],1)
        x = self.activation(self.bn0(self.conv0(x)))

        a1 = self.activation(self.bn2(self.conv2(x)))
        a1 = self.va1(a1)

        x = self.layer3(x)
        a2 = self.activation(self.bn3(self.conv3(x)))
        a2 = self.va2(a2)

        x = self.layer4(x)
        a3 = self.va3(x)
        return [a1,a2,a3]
    
if __name__ == '__main__':
    mspn = ResNet(3, resnet_type='18')
    imgs = torch.randn(3, 2, 160, 200)
    imgs2 = torch.randn(3, 2, 160, 200)
    x = mspn(imgs, imgs2)
    print(x.shape)