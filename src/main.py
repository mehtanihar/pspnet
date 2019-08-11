# Import statements
import numpy as np
import torch
import visdom
import argparse
import configparser
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from PIL import Image
from torch.utils.data import DataLoader
import torch.optim as toptim
from torchvision import transforms
from ignite.engine import Engine, Events, create_supervised_evaluator
import os.path as osp
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, Precision, Recall, MetricsLambda
import init_paths
from models.network import PSPNet,FCN8,SegNet
from models.criterion import CrossEntropyLoss2d
from datasets import helper as dhelper
# from src import utils as myutils
from IPython.core.debugger import set_trace
from argparse import ArgumentParser
from utils import *
from accuracy_metrics import pixel_accuracy,mean_accuracy,mean_IU

# Global variables
NUM_CHANNELS = 3
NUM_CLASSES = 22  #6 for brats

def create_plot_window(vis, xlabel, ylabel, title, win, env, trace_name):
    if not isinstance(trace_name, list):
        trace_name = [trace_name]
    print(xlabel, ylabel, title)
    vis.line(X=np.array([1]), Y=np.array([np.nan]), win=win, env=env,
    name=trace_name[0],
    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
    for name in trace_name[1:]:
        vis.line(X=np.array([1]), Y=np.array([np.nan]), win=win, env=env,
        name=name)

def get_im_transform(stats_filename, dataset_name, im_height, im_width):
	stats_filename = osp.join('..', 'data', dataset_name, stats_filename)
	mean_p, std_p = np.loadtxt(stats_filename)
	im_transform = transforms.Compose([
	transforms.Resize(size=(im_height, im_width)),
	transforms.ToTensor(),
	transforms.Normalize(mean_p, std_p)])
	target_transform = transforms.Compose([
	transforms.Resize(size=(im_height, im_width)),
	ToLabel(),
    Relabel(255, 21),
	])
	return im_transform, target_transform

def train(dataset, model_name, config, experiment_suffix, device, checkpoint_filename=None):

    section = config['hyperparams']
    num_classes = int(section['num_classes'])
    im_width = int(section['im_width'])
    im_height = int(section['im_height'])

    section = config['optim']
    base_lr = float(section['base_lr'])
    base_nm = float(section['base_nm'])
    base_wd = float(section['base_wd'])
    max_epochs = int(section['max_epochs'])
    batch_size = int(section['batch_size'])
    shuffle = bool(int(section['shuffle']))

    section = config['misc']
    log_interval = int(section['log_interval'])
    visdom_server = section['visdom_server']
    num_workers = int(section['num_workers'])
    val_interval = int(section['val_interval'])
    dataset_name = dataset
    do_checkpoint = bool(int(section['do_checkpoint']))

    weight = torch.ones(num_classes)
    weight[0] = 0

    def get_dataloader(im_transforms, target_transforms, train):
    	dset = getattr(dhelper, '{:s}Dataset'.format(dataset_name))(
    	input_transform = im_transforms, target_transform = target_transforms,
        train = train)
    	dloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
    	  pin_memory=True, num_workers=num_workers)
    	return dset, dloader

    # train data loader
    im_transforms, target_transforms = get_im_transform("stats.txt",
     dataset_name, im_height, im_width)
    train_dset, train_dloader = get_dataloader(im_transforms,
     target_transforms, True)

    # val data loader
    im_transforms, target_transforms = get_im_transform("stats.txt",
     dataset_name, im_height, im_width)
    val_dset, val_dloader = get_dataloader(im_transforms, target_transforms,
    False)

    # model
    pretrained = True
    if checkpoint_filename is not None:
    	cnn_name = checkpoint_filename.split('/')[-1].split('.')[0].split('_')[1]
    	try:
    		cnn = globals()[cnn_name]
    	except KeyError:
    		print('Could not find CNN implementation for {:s}'.format(cnn_name))
    		raise NotImplementedError
    if(model_name.startswith("PSP")):
        model = PSPNet(num_classes=num_classes, pretrained=pretrained)
        optim = toptim.SGD(model.parameters(), lr=base_lr, momentum = base_nm,
        weight_decay = base_wd)

    elif(model_name.startswith("FCN")):
        model = FCN8(num_classes=num_classes)
        optim = toptim.SGD(model.parameters(), 1e-4, .9, 2e-5)
    elif(model_name.startswith("SegNet")):
        model = SegNet(num_classes=num_classes)
        optim = toptim.SGD(model.parameters(),  1e-3, .9)

    # load checkpoint
    if checkpoint_filename is not None:
    	checkpoint = torch.load(checkpoint_filename)
    	model.load_state_dict(checkpoint.state_dict())
    model.to(device=device)

    # optimizer
    # loss function
    train_loss_fn = CrossEntropyLoss2d(weight)
    train_loss_fn.to(device=device)
    val_loss_fn = CrossEntropyLoss2d(weight)
    val_loss_fn.to(device=device)

    # engines
    def train_loop(engine: Engine, batch):
        ims, target = batch
        #ims = torch.stack(ims).to(device=device, non_blocking=True)
        ims = ims.to(device = device, non_blocking = True)
        target = target.to(device = device, non_blocking = True)
        model.train()
        optim.zero_grad()
        output = model(ims)

        loss = train_loss_fn(output, target)
        loss.backward()
        optim.step()

        train_acc = [pixel_accuracy(output[i].cpu().data.max(0)[1].detach(),
        target[i,0,:,:].cpu().data.detach()) for i in range(len(output))]
        train_acc = sum(train_acc) / len(train_acc)
        train_IU = [mean_IU(output[i].cpu().data.max(0)[1].detach(),
         target[i,0,:,:].cpu().data.detach(), num_classes,  ignore_index = [num_classes - 1]) for i in range(len(output))]
        train_IU = sum(train_IU) / len(train_IU)
        return loss.item(), train_acc, train_IU
    trainer = Engine(train_loop)

    def val_loop(engine: Engine, batch):
        ims, target = batch
        #ims = torch.stack(ims).to(device=device, non_blocking=True)
        ims = ims.to(device = device, non_blocking = True)
        target = target.to(device = device, non_blocking = True)

        model.eval()
        with torch.no_grad():
          output = model(ims)
          loss = val_loss_fn(output, target)

        val_acc = [pixel_accuracy(output[i].cpu().data.max(0)[1].detach(),
        target[i,0,:,:].cpu().data.detach()) for i in range(len(output))]
        val_acc = sum(val_acc) / len(val_acc)
        val_IU = [mean_IU(output[i].cpu().data.max(0)[1].detach(),
        target[i,0,:,:].cpu().data.detach(), num_classes, ignore_index = [num_classes - 1]) for i in range(len(output))]
        val_IU = sum(val_IU) / len(val_IU)
        return loss.item(), val_acc, val_IU
    valer = Engine(val_loop)


    # timers
    train_timer = Timer(average=True)
    train_timer.attach(trainer,
    start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
    pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
    val_timer = Timer(average=True)
    val_timer.attach(valer,
    start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
    pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # callbacks
    vis = visdom.Visdom(server=visdom_server)
    exp_name = '{:s}_{:s}'.format(dataset_name,
    model_name)
    if experiment_suffix:
    	exp_name += '_{:s}'.format(experiment_suffix)
    print('Starting experiment {:s}'.format(exp_name))
    loss_win = 'loss'
    create_plot_window(vis, '#Epochs', 'Loss', 'Training and Validation Loss',
    win=loss_win, env=exp_name, trace_name=['train_loss', 'val_loss'])

    acc_win = 'acc'
    create_plot_window(vis, '#Epochs', 'Accuracy',
     'Training and Validation Accuracy',
    win=acc_win, env=exp_name, trace_name=['train_acc', 'val_acc'])

    lr_win = 'lr'
    create_plot_window(vis, '#Epochs', 'log10(lr)', 'Learning Rate',
    win=lr_win, env=exp_name, trace_name='lr')

    IU_win = 'IU'
    create_plot_window(vis, '#Epochs', 'IU Accuracy',
     'Training and Validation IU Accuracy',
    win=IU_win, env=exp_name, trace_name=['train_IU', 'val_IU'])

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
    	iter = (engine.state.iteration - 1) % len(train_dloader) + 1

    	if iter % log_interval == 0:
    		print("{:s} train Epoch[{:03d}/{:03d}] Iteration[{:04d}/{:04d}] "
    			"Loss: {:02.2f} Batch time: {:.4f}".format(exp_name,
    			engine.state.epoch, max_epochs, iter, len(train_dloader),
    			engine.state.output[0], train_timer.value()))
    		epoch = engine.state.epoch - 1 +\
    			float(iter-1)/(len(train_dloader)-1)

    		vis.line(X=np.array([epoch]), Y=np.array([engine.state.output[0]]),
    			update='append', win=loss_win, env=exp_name, name='train_loss')
    		vis.line(X=np.array([epoch]), Y=np.array([np.log10(base_lr)]),
    			update='append', win=lr_win, env=exp_name, name='lr')

    @valer.on(Events.ITERATION_COMPLETED)
    def avg_loss_callback(engine: Engine):
        if not hasattr(engine.state, 'avg_val_loss'):
        	engine.state.avg_val_loss = 0
        if not hasattr(engine.state, 'avg_val_IU'):
            engine.state.avg_val_IU = 0
        if not hasattr(engine.state, 'avg_val_acc'):
        	engine.state.avg_val_acc = 0
        it = (engine.state.iteration - 1) % len(train_dloader)

        engine.state.avg_val_loss = (engine.state.avg_val_loss*it + engine.state.output[0]) / \
        						(it + 1)
        engine.state.avg_val_acc = (engine.state.avg_val_acc*it + engine.state.output[1]) / \
        						(it + 1)
        engine.state.avg_val_IU = (engine.state.avg_val_IU*it + engine.state.output[2]) / \
        						(it + 1)

        if it % log_interval == 0:
        	print("{:s} val Iteration[{:04d}/{:04d}] Loss: {:02.2f} \
             Acc: {:02.2f} IU: {:02.2f} Batch time: {:.4f}"
        	.format(exp_name, it, len(val_dloader), engine.state.output[0],
        	engine.state.output[1], engine.state.output[2], val_timer.value()))

    @trainer.on(Events.ITERATION_COMPLETED)
    def avg_loss_callback(engine: Engine):
        if not hasattr(engine.state, 'avg_train_loss'):
        	engine.state.avg_train_loss = 0
        if not hasattr(engine.state, 'avg_train_IU'):
        	engine.state.avg_train_IU = 0
        if not hasattr(engine.state, 'avg_train_acc'):
        	engine.state.avg_train_acc = 0
        it = (engine.state.iteration - 1) % len(train_dloader)

        engine.state.avg_train_loss = (engine.state.avg_train_loss*it + engine.state.output[0]) / \
        						(it + 1)
        engine.state.avg_train_acc = (engine.state.avg_train_acc*it + engine.state.output[1]) / \
        						(it + 1)
        engine.state.avg_train_IU = (engine.state.avg_train_IU*it + engine.state.output[2]) / \
        						(it + 1)

        if it % log_interval == 0:
        	print("{:s} train Iteration[{:04d}/{:04d}] Loss: {:02.2f} \
             Acc: {:02.2f} IU: {:02.2f} Batch time: {:.4f}"
        	.format(exp_name, it, len(train_dloader), engine.state.output[0],
        	engine.state.output[1], engine.state.output[2], train_timer.value()))

    @valer.on(Events.EPOCH_COMPLETED)
    def log_val_loss(engine: Engine):
        vis.line(X=np.array([engine.state.epoch]),
        	Y=np.array([engine.state.avg_val_loss]), update='append', win=loss_win,
        	env=exp_name, name='val_loss')
        vis.line(X=np.array([trainer.state.epoch]),
        	Y=np.array([engine.state.avg_val_acc]), update='append', win=acc_win,
        	env=exp_name, name='val_acc')
        vis.line(X=np.array([trainer.state.epoch]),
        	Y=np.array([engine.state.avg_val_IU]), update='append', win=IU_win,
        	env=exp_name, name='val_IU')

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_val(engine: Engine):
        vis.line(X=np.array([engine.state.epoch]),
        	Y=np.array([engine.state.avg_train_loss]), update='append', win=loss_win,
        	env=exp_name, name='train_loss')
        vis.line(X=np.array([trainer.state.epoch]),
        	Y=np.array([engine.state.avg_train_acc]), update='append', win=acc_win,
        	env=exp_name, name='train_acc')
        vis.line(X=np.array([trainer.state.epoch]),
        	Y=np.array([engine.state.avg_train_IU]), update='append', win=IU_win,
        	env=exp_name, name='train_IU')
        vis.save([exp_name])
        if val_interval < 0:  # don't do validation
            return
        if engine.state.epoch % val_interval != 0:
            return

        valer.run(val_dloader)

    @trainer.on(Events.STARTED)
    def init_trainer_state(engine: Engine):
    	return

    # handlers
    def checkpoint_fn(val_engine: Engine):
    	l = val_engine.state.avg_loss
    	val_engine.state.avg_loss = 0
    	return -l
    checkpoint_handler =\
    	ModelCheckpoint(dirname=osp.join('..', 'data', dataset_name, 'checkpoints'),
    		filename_prefix=exp_name, score_function=checkpoint_fn,
    		score_name='val_loss', create_dir=False, require_empty=False)
    if do_checkpoint:
    	valer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler,
    		{'model': model})

    # RUN!
    trainer.run(train_dloader, max_epochs)

def evaluate(checkpoint_filename, dataset_name, model_name, config, device,
    image_name, label_name):

    section = config['hyperparams']
    im_width = int(section['im_width'])
    im_height = int(section['im_height'])

    num_classes = int(section['num_classes'])
    # model
    pretrained = True
    if(model_name.startswith("PSP")):
        model = PSPNet(num_classes=num_classes, pretrained=pretrained)
    elif(model_name.startswith("FCN")):
        model = FCN8(num_classes=num_classes)
    elif(model_name.startswith("SegNet")):
        model = SegNet(num_classes=num_classes)

    # load checkpoint
    if checkpoint_filename is not None:
    	checkpoint = torch.load(checkpoint_filename)
    	model.load_state_dict(checkpoint.state_dict())
    model.to(device=device)

    dir="../data/VOC12/SegmentationClass"
    ref_image=Image.open(dir+"/2007_000032.png")
    input_transform, target_transform = get_im_transform("stats.txt",
    dataset_name, im_height, im_width)

    im1 = input_transform(Image.open(image_name).convert('RGB')) #240,240
    input_transform1 = transforms.Compose([
        transforms.Resize(size=(im_height, im_width))
    ])
    color_transform = Colorize()
    image_transform = transforms.ToPILImage()
    im2=input_transform1(Image.open(image_name))

    label = model(Variable(im1, volatile=True).unsqueeze(0))#1,N,240,240
    label = color_transform(label[0].data.max(0)[1])#1,3,240,240
    output = image_transform(label)
    output = output.quantize(palette=ref_image)
    output.save(label_name)
    im2=image_transform(im2)
    im2.save(image_name + "_cropped.jpg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required = True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint filename')
    parser.add_argument('--config', type=str, help='Configuration file',
    required=True)
    parser.add_argument('--exp_suffix')
    parser.add_argument('--device', choices=('cpu', 'cuda:0', 'cuda:1'),
    default='cuda:0')
    parser.add_argument('--image')
    parser.add_argument('--label')

    args = parser.parse_args()

    # read the hyperparams
    config = configparser.ConfigParser()
    config.read(args.config)

    if args.train:
    	kwargs = {}
    	if args.checkpoint is not None:
    		kwargs['checkpoint_filename'] = osp.expanduser(args.checkpoint)
    	train(args.dataset, args.model, config, args.exp_suffix,
    		device=args.device, **kwargs)
    else:
        evaluate(args.checkpoint, args.dataset, args.model, config, args.device,
         args.image, args.label)
