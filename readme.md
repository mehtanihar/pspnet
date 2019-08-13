Pytorch implementation of FCN8 and PSPNet

Download trained Deeplab Resnet model from:
https://drive.google.com/open?id=0BxhUwxvLPO7TeXFNQ3YzcGI4Rjg

FCN8 and PSPNet were trained on the PASCAL VOC 2012 Dataset and the BRATS 2017 Dataset.

Results:

FCN8:

Validation: Pixel accuracy: 83%, Mean IU: 43%

<table>
<tr>
<td><img src="https://github.com/mehtanihar/pspnet/blob/master/results/FCN%20VOC%20train/loss.png" width="300"></td>
<td><img src="https://github.com/mehtanihar/pspnet/blob/master/results/FCN%20VOC%20train/acc.png" width="300"></td>
<td><img src="https://github.com/mehtanihar/pspnet/blob/master/results/FCN%20VOC%20train/mean_IU.png" width="300"></td>
</tr>
</table>

PSPNet: 

Validation: Pixel accuracy: 82%, Mean IU: 62%

<table>
<tr>
<td><img src="https://github.com/mehtanihar/pspnet/blob/master/results/PSPNet%20VOC/loss.png" width="300"></td>
<td><img src="https://github.com/mehtanihar/pspnet/blob/master/results/PSPNet%20VOC/acc.png" width="300"></td>
<td><img src="https://github.com/mehtanihar/pspnet/blob/master/results/PSPNet%20VOC/mean_IU.png" width="300"></td>
</tr>
</table>

Setup:
Install the environment using:
```
conda env create -f environment.yml
```

Activate the environment:
```
source activate psp_env
```

Start visdom on port 8097:
```
visdom
```

Training:
```
python main.py [-h] --dataset DATASET --model MODEL [--train] [--test]
                    [--checkpoint CHECKPOINT] --config CONFIG
                    [--exp_suffix EXP_SUFFIX] [--device {cpu,cuda:0,cuda:1}]
                    [--image IMAGE] [--label LABEL]
```

Example: 
```
python main.py --dataset VOC12 --model PSPNet --train --config ../configs/config.ini --exp_suffix 1 --device cpu
```

Validation:
```
python main.py --dataset VOC12 --model PSPNet --test --checkpoint <checkpoint_file>  --config ../configs/config.ini --exp_suffix 1 --image input_image --label output_image
```

References:

https://github.com/bodokaiser/piwise

https://github.com/isht7/pytorch-deeplab-resnet
