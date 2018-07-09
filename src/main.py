import numpy as np
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from transforms import Compose, CenterCrop, Normalize,Scale
from transforms import ToTensor, ToPILImage

import network,dataset,criterion,transform
from dataset import VOC12
from network import PSPNet,FCN8,SegNet,FCN8s
from criterion import CrossEntropyLoss2d
from transform import Relabel, ToLabel, Colorize
import deeplab_resnet
import torch.nn.functional as F
from accuracy_metrics import pixel_accuracy,mean_accuracy,mean_IU

NUM_CHANNELS = 3
NUM_CLASSES = 22  #6 for brats


color_transform = Colorize()
image_transform = ToPILImage()
input_transform = Compose([
	CenterCrop(256),
	#Scale(240),
	ToTensor(),
	Normalize([.485, .456, .406], [.229, .224, .225]),
])

input_transform1 = Compose([
	CenterCrop(256),
	#Scale(136),
	#ToTensor(),
])

target_transform = Compose([
	CenterCrop(256),
	#Scale(240),
	ToLabel(),
	#Relabel(255, NUM_CLASSES-1),
])

target_transform1 = Compose([
	#CenterCrop(256),
	#Scale(136),
	ToLabel(),
	#Relabel(255, NUM_CLASSES-1),
])

def train(args, model):
	model.train()

	weight = torch.ones(NUM_CLASSES)
	weight[0] = 0

	loader = DataLoader(VOC12(args.datadir, input_transform, target_transform),
		num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

	if args.cuda:
		criterion = CrossEntropyLoss2d(weight.cuda())
		#criterion=torch.nn.BCEWithLogitsLoss()
	else:
		criterion = CrossEntropyLoss2d(weight)

	if args.model.startswith('FCN'):
		optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)
	if args.model.startswith('PSP'):
		optimizer=SGD(filter(lambda p: p.requires_grad, model.parameters()), 1e-2,0.9,1e-4)
		#optimizer = SGD(model.parameters(), 1e-2, .9, 1e-4)
	if args.model.startswith('Seg'):
		optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), 1e-3, .9)

	print("Total images:",len(loader))
	best_loss=100
	
	f=open("loss.txt","a")
	for epoch in range(1, args.num_epochs+1):
		epoch_loss = []
		iteration=1
		for step, (images, labels) in enumerate(loader):
			print("Iter:"+str(iteration))
			iteration=iteration+1
			if args.cuda:
					
				images = images.cuda()
				labels = labels.cuda()
			
		   	
			inputs = Variable(images)
			targets = Variable(labels)
			
			outputs=model(inputs)
			
			optimizer.zero_grad()
			
			loss = criterion(outputs, targets[:, 0])
			
			loss.backward()
			optimizer.step()
			print(loss.data[0])
			epoch_loss.append(loss.data[0])
				
			if args.steps_loss > 0 and step>0 and step % args.steps_loss == 0:
				average = sum(epoch_loss) / len(epoch_loss)
				
				epoch_loss=[]						
				if best_loss>average:
					best_loss=average
					torch.save(model.state_dict(), "model_pspnet_VOC_2012_analysis.pth")
					print("Model saved!")
					
				f.write("loss: "+str(average)+" epoch: "+str(epoch)+", step: "+str(step)+"\n")
				f.write("best loss: "+str(best_loss)+" epoch: "+str(epoch)+", step: "+str(step)+"\n")
				print("loss: "+str(average)+" epoch: "+str(epoch)+", step: "+str(step))
				print("best loss: "+str(best_loss)+" epoch: "+str(epoch)+", step: "+str(step))
			print("Best loss: "+str(best_loss))


def evaluate(args, model):
	dir="../data/VOC2012/SegmentationClass"
	ref_image=Image.open(dir+"/2007_000032.png")
	im1 = input_transform(Image.open(args.image).convert('RGB')) #240,240
	im2=input_transform1(Image.open(args.image))
	
	label = model(Variable(im1, volatile=True).unsqueeze(0))#1,N,240,240
	label = color_transform(label[0].data.max(0)[1])#1,3,240,240
	output=image_transform(label)
	output=output.quantize(palette=ref_image)
	output.save(args.label)
	#im2=image_transform(im2)
	#im2.save("cropped.jpg")
	
	
	
def main(args):
	Net = None

	if(args.model =="FCN8"):
		Net=FCN8
	elif(args.model=="SegNet"):
		Net=SegNet
	elif(args.model=="PSPNet"):
		Net=PSPNet
	elif(args.model =="FCN8s"):
		Net=FCN8s
	assert Net is not None, 'model {args.model} not available'

	model = Net(NUM_CLASSES)
	
	if args.cuda:
		#model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
		model = model.cuda()
	if args.state:
		try:
			model.load_state_dict(torch.load(args.state))
		except AssertionError:
			model.load_state_dict(torch.load(args.state,
				map_location=lambda storage, loc: storage))

	if args.mode == 'eval':
		evaluate(args, model)
	if args.mode == 'train':
		train(args, model)

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--model', required=True)
	parser.add_argument('--state')

	subparsers = parser.add_subparsers(dest='mode')
	subparsers.required = True

	parser_eval = subparsers.add_parser('eval')
	parser_eval.add_argument('image')
	parser_eval.add_argument('label')

	parser_train = subparsers.add_parser('train')
	parser_train.add_argument('--port', type=int, default=80)
	parser_train.add_argument('--datadir', required=True)
	parser_train.add_argument('--num-epochs', type=int, default=50)
	parser_train.add_argument('--num-workers', type=int, default=4)
	parser_train.add_argument('--batch-size', type=int, default=1)
	parser_train.add_argument('--steps-loss', type=int, default=100)


	main(parser.parse_args())