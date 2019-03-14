import pickle
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import os

idx_to_label = {}
label_to_idx = {}

def y_to_one_hot(Y, vec_size):
	one_hot_vec = list()
	for y in Y:
		target = [0 for _ in range(vec_size)]
		target[y] = 1
		one_hot_vec.append(target)

	return np.array(one_hot_vec)

def plot_example(X):
	imgplot = plt.imshow(X[:,:,0])
	plt.show()

def data_load(split=0.7, filenames=["ARIAL"]):
	global idx_to_label
	global label_to_idx

	cwd = os.getcwd()
	fontsPath = cwd+"/fonts2"

	filenames = filter(None, filenames)

	#data = pd.DataFrame()
#	for name in filenames:
#		print(name)
#		newData = pd.read_csv(fontsPath+"/"+name+".csv")
#		data = pd.concat([data,newData])

	data = pd.concat([pd.read_csv(fontsPath+"/"+name+".csv") for name in filenames])

	idx_to_label = {idx:name for idx,name in enumerate(data.font.unique())}
	label_to_idx = dict([[v,k] for k,v in idx_to_label.items()])

	num_of_classes = len(label_to_idx)
	print("number of classes: " + num_of_classes)

	X = data.iloc[:,12:].values
	Y = y_to_one_hot([label_to_idx[value] for value in data['font'].values], num_of_classes)

	X = np.true_divide(X,255)

	splitpoint = int(math.floor(len(X)*split))
	X_train = X[:splitpoint]
	X_test = X[splitpoint:]

	Y_train = Y[:splitpoint]
	Y_test = Y[splitpoint:]

	X_train = np.reshape(X_train,(-1,20,20,1))
	X_test = np.reshape(X_train,(-1,20,20,1))

	print(len(X_train),len(Y_train))

	pickle_out = open("idx_to_label.pickle","wb")
	pickle.dump(idx_to_label, pickle_out)
	pickle_out.close()
	return X_test,X_train,Y_test,Y_train,idx_to_label,label_to_idx
