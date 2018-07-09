import numpy as np
from PIL import Image


def pixel_accuracy(image1,image2):
	image1=np.array(image1)	
	image2=np.array(image2)
	[row,col]=image1.shape
	image1=np.reshape(image1,(row*col,1))
	image2=np.reshape(image2,(row*col,1))
	count=0
	total_count=0
	for i in range(row*col):
			total_count+=1
			if(image1[i]==image2[i]):
				count+=1
		
	return count/total_count
	
def mean_accuracy(image1,image2,num_classes):
	image1=np.array(image1)	
	image2=np.array(image2)
	[row,col]=image1.shape
	correct_labels=np.zeros((num_classes,1))
	incorrect_labels=np.zeros((num_classes,1))
	image1=np.reshape(image1,(row*col,1))
	image2=np.reshape(image2,(row*col,1))
	for i in range(row*col):
		if(image1[i]==image2[i]):
			correct_labels[image2[i]]+=1
		else:
			incorrect_labels[image2[i]]+=1
	return ((sum(correct_labels/(correct_labels+incorrect_labels+1e-8)))[0]/sum((correct_labels+incorrect_labels)>0)[0]);
	

def mean_IU(image1,image2,num_classes):
	image1=np.array(image1)	
	image2=np.array(image2)
	[row,col]=image1.shape
	correct_predictions=np.zeros((num_classes,1))
	incorrect_predictions=np.zeros((num_classes,1))
	correct_labels=np.zeros((num_classes,1))
	incorrect_labels=np.zeros((num_classes,1))
	image1=np.reshape(image1,(row*col,1))
	image2=np.reshape(image2,(row*col,1))
	for i in range(row*col):
		if(image1[i]==image2[i]):
			correct_predictions[image1[i]]+=1
			correct_labels[image1[i]]+=1
		else:
			incorrect_predictions[image1[i]]+=1
			incorrect_labels[image2[i]]+=1
	
	return ((sum(correct_predictions/(correct_predictions+incorrect_predictions+incorrect_labels+1e-8)))[0] 
			/sum((correct_predictions+incorrect_predictions+incorrect_labels)>0)[0])
	

#im1=np.array([[0,0,0],[1,1,1]])
#im2=np.array([[1,0,1],[0,0,0]])
#print(mean_accuracy(im1,im2,3))