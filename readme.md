Pytorch implementation of FCN8 and PSPNet

Download trained Deeplab Resnet model from:
https://drive.google.com/open?id=0BxhUwxvLPO7TeXFNQ3YzcGI4Rjg

FCN8 and PSPNet were trained on the PASCAL VOC 2012 Dataset and the BRATS 2017 Dataset.

Results:

FCN8: 
Training: Pixel accuracy: 93%, Mean IU: 62%

Validation: Pixel accuracy: 83%, Mean IU: 43%

PSPNet: Training accuracy: 96%, Validation accuracy: 62%

Training:
python main.py --cuda --model PSPNet train --datadir data --num-epochs 30 --num-workers 12 --batch-size 1

Validation:
python main.py --model PSPNet --state {saved_model_file} eval {input_image} {output_image}

References:

https://github.com/bodokaiser/piwise

https://github.com/isht7/pytorch-deeplab-resnet
