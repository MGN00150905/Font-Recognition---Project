import pandas as pd
import math
import numpy as np
import os
import h5py
from utils import data_load, plot_example
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm


cwd = os.getcwd()
fontsPath = cwd+"/fonts"

fonts = ["AGENCY", "TIMES", "VERDANA", "ARIAL"]

data = pd.read_csv(fontsPath+"/AGENCY.csv")

X_train,X_test,Y_train,Y_test,idx_to_label,label_to_idx = data_load(0.7,fonts)

def get_model(target_shape=153):
	model_name = "model_with_target_"+str(target_shape)+".h5"

	# So it doesnt overide the same model each time
	if os.path.exists(model_name):
		return load_model(model_name)

	else:
		X_input = Input(shape=(20,20,1,))

		conv = Conv2D(32, (3, 3), activation='relu', padding='same')(X_input)

		pool = MaxPooling2D(pool_size=(2, 2))(conv)

		flat = Flatten()(pool)

		dense1 = Dense(512, activation='relu',  kernel_constraint=maxnorm(3))(flat)

		dropout = Dropout(0.5)(dense1)

		dense2 = Dense((target_shape,), activation='softmax')(dropout)

		model = Model(inputs=X_input, outputs=dense2)

		model.compile(loss = 'categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

		model.fit(X_train, Y_train, epochs=10, batch_size=32)

		# Save the model
		model.save(model_name)
		return(model)

model = get_model(len(idx_to_label))
