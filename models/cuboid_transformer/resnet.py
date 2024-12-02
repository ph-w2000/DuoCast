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
from .attention import Transformer3DModel

class VA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.transformer = Transformer3DModel(in_channels=channels)

    def forward(self, x):
        return self.transformer(x)
    


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x
    
class Channel_Spatial(nn.Module):

    def __init__(self, channels , *args, **kwargs):
        super(Channel_Spatial, self).__init__()
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1, 1)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(1, 1)
        self.softmax = nn.Softmax(dim=1)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.relu2 = nn.ReLU()
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x):
        identity = x
        x = self.pooling(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        xc = identity*x + identity
        
        x = self.conv2(xc)
        x = self.relu2(x)
        x = self.softmax2(x)
        
        x = xc*x + xc
       
        
        return x
        
class Res2NetBottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=True,  norm_layer=True):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer:
            norm_layer = nn.BatchNorm2d

        bottleneck_planes = groups * planes
        self.scales = scales
        self.stride = stride
        self.downsample = downsample
        self.pyramid = 4
        
        #1*1的卷积层,在第二个layer时缩小图片尺寸
        self.bn1 = nn.BatchNorm2d(bottleneck_planes)
        self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        #3*3的卷积层，一共有3个卷积层和3个BN层
        self.bn2 =  nn.ModuleList([norm_layer(bottleneck_planes // (scales+self.pyramid)) for _ in range(scales-1)])
        self.conv2 = nn.ModuleList([nn.Conv2d(bottleneck_planes // (scales+self.pyramid), bottleneck_planes // (scales+self.pyramid),
                                              kernel_size=3, stride=1, padding=1, groups=groups, bias=False) for _ in range(scales-1)])
        
        
        self.bn2_pyramid =  nn.ModuleList([norm_layer(bottleneck_planes // (scales+self.pyramid)) for i in range(self.pyramid)])
        self.conv2_pyramid = nn.ModuleList([nn.Conv2d(bottleneck_planes // (scales+self.pyramid), bottleneck_planes // (scales+self.pyramid),
                                              kernel_size=2*i+3, stride=1, padding=i+1, groups=groups, bias=False) for i in range(self.pyramid)])
        
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * self.expansion) if se else None


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out) 
        
        xs = torch.chunk(out, self.scales+self.pyramid, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        y_pyramid = []
        for s in range(self.pyramid):
            y_pyramid.append(self.relu(self.bn2_pyramid[s](self.conv2_pyramid[s](xs[self.scales+s]))))
        out_pyramid = torch.cat(y_pyramid, 1)
        
        out = torch.cat((out,out_pyramid),1)
        
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample:
            identity = self.downsample(identity)
            
        out += identity
        out = self.relu(out)

        return out



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

RESNET_CONFIGS = {'18': [[2, 2, 2, 2], Res2NetBottleneck],
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

class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim ):
        super(ResNet, self).__init__()

        layers, block = RESNET_CONFIGS['18']
        self.activation = nn.ReLU()
        self._norm_layer = nn.BatchNorm2d

        self.in_planes = in_dim
        self.layer1 = self._make_layer(block, out_dim, layers[0], stride=1)
        self.layer2 = self._make_layer(block, out_dim*2, layers[0], stride=2)


        self.va1 = VA(out_dim)
        self.va2 = VA(out_dim*2)
        
        self.initialize_params()



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
        layers.append(block(self.in_planes, planes, downsample, stride=stride, scales=4, groups=1, se=True, norm_layer=norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, scales=4, groups=1, se=True, norm_layer=norm_layer))
           
        return nn.Sequential(*layers)

    def forward(self, x):
        b, t, _, _, _ = x.shape
        x = rearrange(x, "b t h w c-> (b t) c h w ")
        x = self.layer1(x)
        a1 = self.va1(rearrange(x, "(b t) c h w -> b c t h w", t=t))
        a1 = rearrange(a1, "b c t h w -> b t h w c")
        x = rearrange(a1, " b t h w c -> (b t) c h w")
        x = self.layer2(x)
        a2 = self.va2(rearrange(x, "(b t) c h w -> b c t h w", t=t))
        a2 = rearrange(a2, "b c t h w -> b t h w c")

        return [a1,a2]
