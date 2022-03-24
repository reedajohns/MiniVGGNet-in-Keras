# Imports
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K

class Architectures:
	@staticmethod
	# Build simple model
	def simple_build_model(width, height, depth, classes):
		# Initialize model as sequential
		model = Sequential()
		# Format shape into tuple
		input_shape = (height, width, depth)

		# This a simple CNN... so let's keep it real simple
		# Add the only CONV => RELU layer
		# The 32, (3, 3) below says have 32 convultion kernels with a 3x3 size
		# "same" means the output layer H and W will be same as input
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=input_shape))
		# Add relu activation
		model.add(Activation("relu"))

		# Add softmax for classification
		# Need to flatten and add dense (FC) layer prior
		model.add(Flatten())
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# Return model
		return model

	@staticmethod
	# Build LeNet model
	def lenet_build_model(width, height, depth, classes):
		# Initialize model as sequential
		model = Sequential()
		# Format shape into tuple
		input_shape = (height, width, depth)

		# Full layout of architecture:
		# INPUT => CONV => TANH => POOL => CONV => TANH => POOL =>
		# 	FC => TANH => FC

		# (1) Add first CONV => RELU => POOL layers
		# "same" means the output layer H and W will be same as input
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape=input_shape))
		# ReLu act
		model.add(Activation("relu"))
		# Pooling to decrease size (in half)
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL layers
		# "same" means the output layer H and W will be same as input
		model.add(Conv2D(50, (5, 5), padding="same"))
		# ReLu act
		model.add(Activation("relu"))
		# Pooling to decrease size (in half)
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# Add fully connected layers
		# Need to flatted from 2D to 1D vectore
		model.add(Flatten())
		# We'll go down to 500 nodes
		model.add(Dense(500))
		#  Add ReLu act
		model.add(Activation("relu"))

		# Narrow down to number of classes
		model.add(Dense(classes))
		# And finally the softmax for classifying
		model.add(Activation("softmax"))

		# Return model
		return model

	@staticmethod
	# Build MiniVGGNet model
	def minivggnet_build_model(width, height, depth, classes):
		# Initialize model as sequential
		model = Sequential()
		# Format shape into tuple
		input_shape = (height, width, depth)
		chanDim = -1

		# Full layout of architecture:
		# Two sets of CONV => RELU => CONV => RELU => POOL layers,
		# followed by a set of FC => RELU => FC => SOFTMAX layers

		# (1) Add first CONV => RELU => CONV => RELU => POOL layers
		# "same" means the output layer H and W will be same as input
		model.add(Conv2D(32, (3, 3), padding="same",
						 input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (2) Add second CONV => RELU => CONV => RELU => POOL layers
		# "same" means the output layer H and W will be same as input
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Add fully connected layers
		# Need to flatted from 2D to 1D vectore
		model.add(Flatten())
		# We'll go down to 512 nodes
		model.add(Dense(512))
		#  Add ReLu act
		model.add(Activation("relu"))
		# Batch normilzation and dropout layers
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		# Narrow down to number of classes
		model.add(Dense(classes))
		# And finally the softmax for classifying
		model.add(Activation("softmax"))

		# Return model
		return model