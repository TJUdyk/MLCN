import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet
from models.cca import CCA
from models.scr import SCR
from models.PMMs import PMMs
from common.utils import euclidean_dist_similarity
from models.conv4 import ConvNet

class RENet(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        self.encoder = ResNet(args=args)
        self.encoder2 = ConvNet()
        self.encoder_dim = 640
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)

        self.scr_module = self._make_scr_layer(planes=[640, 64, 64, 64, 640])
        self.cca_module = CCA(kernel_sizes=[3, 3], planes=[16, 1])
        self.cca_1x1 = nn.Sequential(
            nn.Conv2d(self.encoder_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.num_pro = 5
        
    def _make_scr_layer(self, planes):
        stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        layers = list()
        self_block = SCR(planes=planes, stride=stride)
        layers.append(self_block)
        return nn.Sequential(*layers)

    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)
        elif self.mode == 'encoder':
            return self.encode(input, False)
        elif self.mode == 'cca':
            spt, qry = input
            return self.cca(spt, qry)
        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        return self.fc(x)


    def encode(self, x, do_gap=True):
        x = self.encoder(x)
        if do_gap:
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x
    
    def cca(self, spt, qry ):
        spt = spt.squeeze(0) 
        way = spt.shape[0]   
        num_qry = qry.shape[0]  

        spt = self.normalize_feature(spt)  
        qry = self.normalize_feature(qry)
        qry_pooled = qry.mean(dim=[-1, -2])

        corr4d = self.get_4d_correlation_map(spt, qry)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()

        corr4d = self.cca_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))

        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_s = F.softmax(corr4d_s / self.args.temperature_attn, dim=2)
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        attn_s = corr4d_s.sum(dim=[4, 5])
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)

        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)
        corr4d_q = F.softmax(corr4d_q / self.args.temperature_attn, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)
        attn_q = corr4d_q.sum(dim=[2, 3])
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)

        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        spt_attended_pooled = spt_attended.mean(dim=[-1, -2])
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2])
        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)

        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry_pooled)
        else:
            return similarity_matrix / self.args.temperature

    def get_4d_correlation_map(self, spt, qry):

        way = spt.shape[0]
        num_qry = qry.shape[0]
        spt = self.cca_1x1(spt)
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)
        
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)

        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        return similarity_map_einsum
    
    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)