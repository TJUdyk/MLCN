import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple
import torch.nn.init as init
from models.attention import  ChannelAttention ,  SpatialAttention


class SCR(nn.Module):
    def __init__(self, planes=[640, 64, 64, 64, 640], stride=(1, 1, 1), ksize=3, do_padding=False, bias=False):
        super(SCR, self).__init__()
        self.ksize = _quadruple(ksize) if isinstance(ksize, int) else ksize
        padding1 = 0

        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(planes[1], planes[2], 3,
                                             stride=1, bias=bias, padding=padding1),
                                   nn.BatchNorm2d(planes[2]),
                                   nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(nn.Conv2d(planes[2], planes[3], 3,
                                             stride=1, bias=bias, padding=padding1),
                                   nn.BatchNorm2d(planes[3]),
                                   nn.ReLU(inplace=False))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[4]))
        self.ca1 = ChannelAttention(planes[0])
        self.sa1 = SpatialAttention()
    
    def forward(self, x):
        b, c, h, w  = x.shape 
        out = x
        out = self.sa1(out) 
        out = self.conv1x1_in(x) 
        out = self.conv1(out) 
        out = self.conv2(out) 
        c = out.shape[1]
        out = self.conv1x1_out(out)
        return out

