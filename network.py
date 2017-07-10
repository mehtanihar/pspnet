import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.utils import model_zoo
from torchvision import models
import deeplab_resnet
from torch.autograd import Variable



class PSPDec(nn.Module):

    def __init__(self, in_features, out_features, downsize, upsize=18):
        super(PSPDec,self).__init__()

        self.features = nn.Sequential(
            nn.AvgPool2d(downsize, stride=downsize),
            nn.Conv2d(in_features, out_features, 1, bias=False),
            nn.BatchNorm2d(out_features, momentum=.95),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(upsize)
        )

    def forward(self, x):
        return self.features(x)


class PSPNet(nn.Module):

    def __init__(self, num_classes):
        super(PSPNet,self).__init__()

        init_net=deeplab_resnet.Res_Deeplab()

        #resnet = models.resnet101(pretrained=True)

        state=torch.load("../models/MS_DeepLab_resnet_trained_VOC.pth")
        init_net.load_state_dict(state)
        self.resnet=init_net
        
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #m.stride = 1
                m.requires_grad = False
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = False
        

        self.layer5a = PSPDec(21, 5, 18)
        self.layer5b = PSPDec(21, 5, 9)
        self.layer5c = PSPDec(21, 5, 6)
        self.layer5d = PSPDec(21, 5, 3)

        self.final = nn.Sequential(
            nn.Conv2d(41, 25, 3, padding=1, bias=False),
            nn.BatchNorm2d(25, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(25, num_classes, 1),
        )

    def forward(self, x):
        
        
        x=self.resnet(x)
        x=x[0]
        
        x = self.final(torch.cat([
            x,
            self.layer5a(x),
            self.layer5b(x),
            self.layer5c(x),
            self.layer5d(x),
        ], 1))

        #print('final', x.size())

        return F.upsample_bilinear(x,136)
