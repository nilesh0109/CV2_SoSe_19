import numpy as np
import os
from neuralNetwork import train_network

"""
unpickle function from CIFAR-10 website
"""
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='latin')
    return d

def create_dataset_from_files(files):
	rawdata = []
	labels = []
	for f in files:
		d = unpickle(f)
		rawdata.extend(d["data"])
		labels.extend(d["labels"])

	rawdata = np.array(rawdata)

	red = rawdata[:,:1024].reshape((-1,32,32))
	green = rawdata[:,1024:2048].reshape((-1,32,32))
	blue = rawdata[:,2048:].reshape((-1,32,32))

	data = np.stack((red,green,blue), axis=3)
	labels = np.array(labels)

	return data, labels

cifar_data_path = 'data/cifar10/cifar-10-batches-py/' # replace with your path
cifar_training_files = [os.path.join(cifar_data_path, 'data_batch_{:d}'.format(i)) for i in range(1,6)]
cifar_testing_files = [os.path.join(cifar_data_path, 'test_batch')]

train_data, train_labels = create_dataset_from_files(cifar_training_files)
test_data, test_labels = create_dataset_from_files(cifar_testing_files)

print(train_data.shape, train_data.dtype) # (50000, 32, 32, 3) -- 50000  images, each 32 x 32 with 3 color channels (UINT8)
print(train_labels.shape, train_labels.dtype) # 50000 labels, one for each image (INT64)

print(test_data.shape, test_data.dtype) # 10000 images, each 32 x 32 with 3 color channels (UINT8)
print(train_labels.shape, train_labels.dtype) # 10000 labels, one for each image (INT64)

train_network(train_data[0:512], train_labels[0:512], 32, 0.001)
