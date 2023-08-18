import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90

class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Here, we have provided an example structure for the weights and biases.
		It is a list of weight and bias matrices, in which, the
		dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
		weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
		biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

		Please note that this is just an example.
		You are free to modify or entirely ignore this initialization as per your need.
		Also you can add more state-tracking variables that might be useful to compute
		the gradients efficiently.


		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''
		self.num_layers = num_layers
		self.num_units = num_units

		self.biases = []
		self.weights = []
		for i in range(num_layers):

			if i==0:
				# Input layer
				self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
			else:
				# Hidden layer
				self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

			self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

		# Output layer
		self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.

		Note that for a classification task, the output layer should
		be a softmax layer. So perform the computations accordingly

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''
		a=X
		h=None
		self.h_states = []
		self.a_states = []

		for i, (w, b) in enumerate(zip(self.weights, self.biases)):
			if i==0:
				self.h_states.append(a)
			else:
				self.h_states.append(h)
			
			self.a_states.append(a)
			
			h = np.dot(a, w) + b.T
			
			if i < len(self.weights)-1:
				a = relu(h)
			else:
				a = h

		self.pred = a

		return self.pred

		raise NotImplementedError

	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing backward pass.
		'''
		del_W = []
		del_b = []
		y_pred = self.pred
		M = X.shape[0]
		L = self.num_layers + 1				#including output layer

		backprop_error = [(y_pred-y)]
		idx_lst = np.arange(2, L+1)[::-1]	#aligning weight metrices with backpropagation error	

		for idx in idx_lst:
			delta = backprop_error[-1].dot(self.weights[idx-1].T)
			delta = delta*relu_diff(self.a_states[idx-1])
			backprop_error.append(delta)

		backprop_error.reverse()			#re-align to compute weights and bias gradients

		for l in range(L):
			del_W += [self.a_states[l].T.dot(backprop_error[l])/M + lamda*self.weights[l]/M]
			d_b = np.mean(backprop_error[l], axis = 0) + lamda*self.biases[l].T/M
			del_b += [d_b.T]

		return del_W, del_b

		raise NotImplementedError

def relu(mat):
	return np.clip(mat, 0, None)

def relu_diff(mat):
	return np.clip(np.sign(mat), 0, None)

class Optimizer(object):
	'''
	'''

	def __init__(self, learning_rate):
		'''
		Create a Gradient Descent based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''
		self.learning_rate = learning_rate

		# raise NotImplementedError

	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		Returns
		----------
		weights_updated, biases_updated
		'''
		weights_updated, biases_updated = [], []
		for i in range(len(weights)):
			new_w = weights[i] - self.learning_rate*delta_weights[i]
			new_b = biases[i] - self.learning_rate*delta_biases[i]

			weights_updated.append(new_w)
			biases_updated.append(new_b)

		return weights_updated, biases_updated

		raise NotImplementedError


class MinMaxScaler(object):
	'''
	performs min max scaling given a set of feature vectors to a given range
	by default the range is [0,1]
	
	Transform to custom range [A, B] (default A = 0, B = 1)
	x_scaled = (x-X.min)/(X.max-X.min)
	x_scaled = x_scaled*(B-A) + A
	
	Inverse operation
	x = ((x_scaled-A)*(X.max-X.min))/(B-A) + X.min
	Scaling Object initialized in constructor call 
	tranform() call transforms and returns the scaled features
	inverse_transform() performs inverse scaling to return the value in the original range
	
	NOTE: No assertion is done to check if X.max=X.min or B>A
	'''

	def __init__(self, feature_mat, low=0, high=1):
		self.low = low
		self.high = high

		self.feature_max = feature_mat.max(axis=0)
		self.feature_min = feature_mat.min(axis=0)
		self.feature_mat = feature_mat

	def transform(self, feature_mat = None):

		if feature_mat is None:
			feature_mat = self.feature_mat

		scaled_features = (feature_mat-self.feature_min)*(self.high - self.low)/(self.feature_max - self.feature_min) + self.low
		self.scaled_features = scaled_features

		return scaled_features

	def inverse_transform(self, scaled_feature_mat = None):

		if scaled_feature_mat is None:
			scaled_features = self.scaled_features

		features_hat = (scaled_features-self.low)*(self.feature_max - self.feature_min)/(self.high - self.low) + self.feature_min

		return features_hat



def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	'''
	m = y.shape[0]
	l = np.sum((y-y_hat)**2)/m

	return l

	raise NotImplementedError

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss 
	'''
	l2_sum = 0
	for i in range(len(weights)):
		l2_sum += np.sum(weights[i]**2)

	return l2_sum

	raise NotImplementedError

def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''

	reg_loss = loss_mse(y, y_hat) + lamda*loss_regularization(weights, biases)

	return reg_loss

	raise NotImplementedError

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	'''
	m = y.shape[0]
	rmse_val = np.sqrt(np.sum((y-y_hat)**2)/m)

	return rmse_val

	raise NotImplementedError

def cross_entropy_loss(y, y_hat):
	'''
	Compute cross entropy loss

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		cross entropy loss
	'''
	raise NotImplementedError
'''
plot the train loss (MSE) and the dev loss (RMSE)
batch_size needs to be passed as parameter for title
generates a line plot for 100 epochs
'''
def loss_plot_viz(epoch, train_loss, dev_loss, batch_size = 32):

	plt.figure()
	title = "Train Loss Plot for Batch Size: " + str(batch_size)
	plt.title(title)
	plt.plot(epoch, train_loss, label = 'Train Loss (MSE)')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()


	plt.figure()
	title = "Dev Loss Plot for Batch Size: " + str(batch_size)
	plt.title(title)
	plt.plot(epoch, dev_loss, label = 'Dev Loss (RMSE)')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()


def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.

	Here we have added the code to loop over batches and perform backward pass
	for each batch in the loop.
	For this code also, you are free to heavily modify it.
	'''

	m = train_input.shape[0]
	train_loss_mse = []
	dev_loss_rmse = []
	epoch_num = []

	for e in range(max_epochs):
		epoch_loss = 0.
		for i in range(0, m, batch_size):
			batch_input = train_input[i:i+batch_size]
			batch_target = train_target[i:i+batch_size]
			pred = net(batch_input)

			# Compute gradients of loss w.r.t. weights and biases
			dW, db = net.backward(batch_input, batch_target, lamda)

			# Get updated weights based on current weights and gradients
			weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

			# Update model's weights and biases
			net.weights = weights_updated
			net.biases = biases_updated

			# Compute loss for the batch
			batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
			epoch_loss += batch_loss

			#print(e, i, rmse(batch_target, pred), batch_loss)

		print('Epoch:', e, 'Loss:', epoch_loss)
		dev_pred = net(dev_input)
		dev_rmse = rmse(dev_target, dev_pred)
		train_pred = net(train_input)
		train_mse = loss_fn(train_target, train_pred, net.weights, net.biases, lamda)

		print('Train Loss:', train_mse, 'Dev Loss:', dev_rmse)
		if e < 100:			#Plot for 100 epochs
			train_loss_mse.append(train_mse)
			dev_loss_rmse.append(dev_rmse)
			epoch_num.append(e)


		# Write any early stopping conditions required (only for Part 2)
		# Hint: You can also compute dev_rmse here and use it in the early
		# 		stopping condition.

	# After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
	# dev_pred = net(dev_input)
	# dev_rmse = rmse(dev_target, dev_pred)
	loss_plot_viz(epoch_num, train_loss_mse, dev_loss_rmse, batch_size)

	# print('RMSE on dev data: {:.5f}'.format(dev_rmse))


def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	raise NotImplementedError

'''
scales features and target
returns scaled data and MinMaxScaler Objects
the objects will help to perform the respective inverse operation
'''

def scale_hq(train_input, dev_input, test_input, train_target, dev_target):

	#concatenate all data along along row i.e. axis=0
	feature = np.concatenate([train_input, dev_input, test_input], axis=0)
	target = np.concatenate([train_target, dev_target], axis=0)

	#scale features
	feature_scaler = MinMaxScaler(feature)
	train_input = feature_scaler.transform(train_input)
	dev_input = feature_scaler.transform(dev_input)
	test_input = feature_scaler.transform(test_input)

	#scale target
	target_scaler = MinMaxScaler(target)
	train_target = target_scaler.transform(train_target)
	dev_target = target_scaler.transform(dev_target)

	return train_input, dev_input, test_input, train_target, dev_target, feature_scaler, target_scaler


def read_data(data_dir):
	'''
	Read the train, dev, and test datasets
	'''
	train_path = os.path.join(data_dir, 'train.csv')
	dev_path = os.path.join(data_dir, 'dev.csv')
	test_path = os.path.join(data_dir, 'test.csv')
	df_train = pd.read_csv(train_path, header=0)
	df_dev = pd.read_csv(dev_path, header=0)
	df_test = pd.read_csv(test_path, header=0)
	
	print(df_train.shape, df_dev.shape, df_test.shape)

	train_target = df_train['1'].to_numpy()
	train_target = train_target.reshape((len(train_target), 1))
	train_input = df_train.iloc[:, 1:].to_numpy()

	dev_target = df_dev['1'].to_numpy()
	dev_target = dev_target.reshape((len(dev_target), 1))
	dev_input = df_dev.iloc[:, 1:].to_numpy()

	test_input = df_test.iloc[ : , : ].to_numpy()

	return train_input, train_target, dev_input, dev_target, test_input

	# raise NotImplementedError


def main():

	# Hyper-parameters 
	max_epochs = 500
	batch_size = 32
	learning_rate = 0.001
	num_layers = 2
	num_units = 64
	lamda = 0.001 # Regularization Parameter

	#take as input the data directory. If given path is not a directory program will exit
	data_dir = input('Enter data directory: ')
	if not os.path.isdir(data_dir):
		sys.exit("Enter proper directory! Check \'readme.txt\' provided.")

	train_input, train_target, dev_input, dev_target, test_input = read_data(data_dir)
	train_input, dev_input, test_input, train_target, dev_target, featureScale, targetScale = scale_hq(train_input, dev_input, test_input, train_target, dev_target)
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	# get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()
