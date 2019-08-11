import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
import torchvision
import torch.nn.functional as F
import numpy as np
from IPython.core.debugger import set_trace

from torch.utils import model_zoo
import deeplab_resnet
from torch.autograd import Variable
import scipy.misc
from PIL import Image



def kmoment(x,k):
    return np.sum((x)**k) / np.size(x)


class FCN8(nn.Module):

	def __init__(self, num_classes):
		super().__init__()

		feats = list(models.vgg16(pretrained=True).features.children())

		self.feats = nn.Sequential(*feats[0:10])
		self.feat3 = nn.Sequential(*feats[10:17])
		self.feat4 = nn.Sequential(*feats[17:24])
		self.feat5 = nn.Sequential(*feats[24:31])

		self.fconn = nn.Sequential(
			nn.Conv2d(512, 4096, 7,padding=3),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Conv2d(4096, 4096, 1),
			nn.ReLU(inplace=True),
			nn.Dropout(),
		)
		self.score_feat3 = nn.Conv2d(256, num_classes, 1)
		self.score_feat4 = nn.Conv2d(512, num_classes, 1)
		self.score_fconn = nn.Conv2d(4096, num_classes, 1)

	def forward(self, x):

		#Size of input=1,num_classes,256,256
		feats = self.feats(x) #1,128,64,64
		feat3 = self.feat3(feats)#1,256,32,32
		feat4 = self.feat4(feat3)#1,512,16,16
		feat5 = self.feat5(feat4)#1,512,8,8
		fconn = self.fconn(feat5)#1,4096,8,8

		score_feat3 = self.score_feat3(feat3)#1,num_classes,32,32
		score_feat4 = self.score_feat4(feat4)#1,num_classes,16,16
		score_fconn = self.score_fconn(fconn)#1,num_classes,8,8

		score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
		score += score_feat4
		score = F.upsample_bilinear(score, score_feat3.size()[2:])
		score += score_feat3

		output = F.upsample_bilinear(score, x.size()[2:])#1,num_classes,256,256

		return output


class PyramidPool(nn.Module):

	def __init__(self, in_features, out_features, pool_size):
		super(PyramidPool,self).__init__()

		self.features = nn.Sequential(
			nn.AdaptiveAvgPool2d(pool_size),
			nn.Conv2d(in_features, out_features, 1, bias=False),
			nn.BatchNorm2d(out_features, momentum=.95),
			nn.ReLU(inplace=True)
		)


	def forward(self, x):
		size=x.size()
		output=F.upsample(self.features(x), size[2:], mode='bilinear')
		return output


class PSPNet(nn.Module):

    def __init__(self, num_classes, pretrained = False):
        super(PSPNet,self).__init__()
        print("initializing model")
        #init_net=deeplab_resnet.Res_Deeplab()
        #state=torch.load("models/MS_DeepLab_resnet_trained_VOC.pth")
        #init_net.load_state_dict(state)
        self.resnet = torchvision.models.resnet50(pretrained = pretrained)


        self.layer5a = PyramidPool(2048, 512, 1)
        self.layer5b = PyramidPool(2048, 512, 2)
        self.layer5c = PyramidPool(2048, 512, 3)
        self.layer5d = PyramidPool(2048, 512, 6)




        self.final = nn.Sequential(
        	nn.Conv2d(4096, 512, 3, padding=1, bias=False),
        	nn.BatchNorm2d(512, momentum=.95),
        	nn.ReLU(inplace=True),
        	nn.Dropout(.1),
        	nn.Conv2d(512, num_classes, 1),
        )

        initialize_weights(self.layer5a,self.layer5b,self.layer5c,self.layer5d,self.final)




    def forward(self, x):
        count=0

        size=x.size()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.final(torch.cat([
        	x,
        	self.layer5a(x),
        	self.layer5b(x),
        	self.layer5c(x),
        	self.layer5d(x),
        ], 1))


        return F.upsample_bilinear(x,size[2:])



class SegNet(nn.Module):

	def __init__(self, num_classes):
		super().__init__()
		self.num_classes=num_classes
		encoders=list(vgg.vgg16_bn(pretrained=True).features.children())
		self.enc1 = nn.Sequential(*encoders[:6])
		self.pool1=nn.Sequential(*encoders[6:7])
		self.enc2 = nn.Sequential(*encoders[7:13])
		self.pool2=nn.Sequential(*encoders[13:14])
		self.enc3 = nn.Sequential(*encoders[14:23])
		self.pool3=nn.Sequential(*encoders[23:24])
		self.enc4 = nn.Sequential(*encoders[24:33])
		self.pool4=nn.Sequential(*encoders[33:34])
		self.enc5 = nn.Sequential(*encoders[34:43])
		self.pool5=nn.Sequential(*encoders[43:44])


		self.pool=nn.MaxPool2d(2, stride=2,dilation=1,return_indices=True);
		self.unpool=nn.MaxUnpool2d(2, stride=2);
		self.dec5=nn.Sequential(
			nn.Conv2d(512, 512, 3, padding=1,dilation=1),
			nn.BatchNorm2d(512,momentum=0.1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1,dilation=1),
			nn.BatchNorm2d(512,momentum=0.1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1,dilation=1),
			nn.BatchNorm2d(512,momentum=0.1),
			nn.ReLU(inplace=True),
		)

		self.dec4=nn.Sequential(
			nn.Conv2d(512, 512, 3, padding=1,dilation=1),
			nn.BatchNorm2d(512,momentum=0.1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1,dilation=1),
			nn.BatchNorm2d(512,momentum=0.1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1,dilation=1),
			nn.BatchNorm2d(512,momentum=0.1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 256, 3, padding=1,dilation=1),
			nn.BatchNorm2d(256,momentum=0.1),
			nn.ReLU(inplace=True),
		)

		self.dec3=nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=1,dilation=1),
			nn.BatchNorm2d(256,momentum=0.1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1,dilation=1),
			nn.BatchNorm2d(256,momentum=0.1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 128, 3, padding=1,dilation=1),
			nn.BatchNorm2d(128,momentum=0.1),
			nn.ReLU(inplace=True),
		)

		self.dec2=nn.Sequential(
			nn.Conv2d(128, 128, 3, padding=1,dilation=1),
			nn.BatchNorm2d(128,momentum=0.1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, 3, padding=1,dilation=1),
			nn.BatchNorm2d(64,momentum=0.1),
			nn.ReLU(inplace=True),
		)

		self.dec1=nn.Sequential(
			nn.Conv2d(64, 64, 3, padding=1,dilation=1),
			nn.BatchNorm2d(64,momentum=0.1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, self.num_classes, 3, padding=1,dilation=1),
			nn.BatchNorm2d(self.num_classes,momentum=0.1),
			nn.ReLU(inplace=True),
		)



	def forward(self, x):

		enc1=self.enc1(x)
		pool1,pool1_indices=self.pool(enc1)
		enc2=self.enc2(pool1)
		pool2,pool2_indices=self.pool(enc2)
		enc3=self.enc3(pool2)
		pool3,pool3_indices=self.pool(enc3)
		enc4=self.enc4(pool3)
		pool4,pool4_indices=self.pool(enc4)
		enc5=self.enc5(pool4)
		pool5,pool5_indices=self.pool(enc5)

		unpool5=self.unpool(pool5,pool5_indices)
		dec5=self.dec5(unpool5)
		unpool4=self.unpool(pool4,pool4_indices)
		dec4=self.dec4(unpool4)
		unpool3=self.unpool(pool3,pool3_indices)
		dec3=self.dec3(unpool3)
		unpool2=self.unpool(pool2,pool2_indices)
		dec2=self.dec2(unpool2)
		unpool1=self.unpool(pool1,pool1_indices)
		dec1=self.dec1(unpool1)

		return dec1


def initialize_weights(*models):
	for model in models:
		for module in model.modules():
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
				nn.init.kaiming_normal(module.weight)
				if module.bias is not None:
					module.bias.data.zero_()
			elif isinstance(module, nn.BatchNorm2d):
				module.weight.data.fill_(1)
				module.bias.data.zero_()
