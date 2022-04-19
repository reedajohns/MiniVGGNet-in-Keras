# USAGE
# python cifar10_minivggnet_cnn.py --output ../media/keras_minivggnet_cnn_cifar.png --model ../models/minivggnet_weights.hdf5
# Import
# set the matplotlib backend to save figures in background
import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from fundamentals.neuralnet import Architectures
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Adjustable Parameters
learning_rate = 0.01
decay = 0.01 / 40
momentum = 0.9
nesterov_bool = True
batch_size = 64
num_epochs = 40
loss_type = "categorical_crossentropy"

# Function to plot trained model loss / accuracy
def plot_loss_accuracy(H, args):
	# Set up plot
	plt.style.use("ggplot")
	plt.figure()
	# Plot
	plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, num_epochs), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, num_epochs), H.history["val_accuracy"], label="val_acc")
	# Titles / labels
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()

	# Save to output
	plt.savefig(args["output"])

# Main
if __name__ == '__main__':
	# Parse commandline arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-o", "--output", required=True,
		help="path to output loss plot")
	ap.add_argument("-m", "--model", type=str, default='',
					help="Path to save model. (doesn't save if not set)")
	args = vars(ap.parse_args())

	# Load the MNIST dataset from keras
	# It comes separated into training / testing sets so that's nice
	((trainX, trainY), (testX, testY)) = cifar10.load_data()

	# Scale to [0, 1]
	trainX = trainX.astype('float') / 255.0
	testX = testX.astype('float') / 255.0

	# One hot encoding for labels
	lb = LabelBinarizer()
	# Training and test labels
	trainY = lb.fit_transform(trainY)
	testY = lb.transform(testY)

	# Build and finish the model by setting optimizer
	print("[INFO] Finishing model...")
	# Build
	model = Architectures.minivggnet_build_model(width=32, height=32, depth=3, classes=10)
	# Finish
	optimizer = SGD(lr = learning_rate, decay=decay, momentum=momentum, nesterov=nesterov_bool)
	model.compile(loss=loss_type, optimizer=optimizer, metrics=["accuracy"])

	# Train Network
	print('[INFO] Training the network:')
	H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batch_size, epochs=num_epochs, verbose=1)

	# Save network model to disk (if set)
	if args["model"] != '':
		print("[INFO] Serializing model...")
		model.save(args["model"])

	# Label names for the dataset
	labelNames = ["airplane", "automobile", "bird", "cat", "deer",
				  "dog", "frog", "horse", "ship", "truck"]

	# Evaluate on the testing set
	print('[INFO] Evaluating the network:')
	predict = model.predict(testX, batch_size=batch_size)
	# Print handy report of results
	print(classification_report(testY.argmax(axis=1), predict.argmax(axis=1), target_names=labelNames))

	# Save model architecture
	filename_plot = args["output"].spli('.png')[0] + '_architecture.png'
	plot_model(model, to_file=filename_plot, show_shapes=True)

	# Plot training loss / accuracy
	plot_loss_accuracy(H, args)


	print('Done')