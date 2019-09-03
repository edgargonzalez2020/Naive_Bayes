import numpy as np
import pandas
import sys
import operator

class NaiveBayes:
	def __init__(self, training_path, test_path):
		self.training_path = training_path
		self.test_path = test_path
		self.classes_train = {}
		self.count = 0
		self.test_data = None
		# The dictionary below contains
		# (mead, stdev, prior prob)
		self.statistics = {}
		self.load_classes(training_path, True)
		self.load_classes(test_path, False)
		self.train()
	def load_data(self, data_path):
		return np.loadtxt(data_path)
	def load_classes(self, data_path, is_training):
		data = self.load_data(data_path)
		if is_training:
			for row in data:
				self.count += 1
				class_label = int(row[-1])
				if class_label not in self.classes_train:
					self.classes_train[class_label] = row[:-1]
				else:
					self.classes_train[class_label] = np.vstack([self.classes_train[class_label], row[:-1]])
			self.classes_train = dict(sorted(self.classes_train.items()))
		else:
			self.test_data = self.load_data(self.test_path)
	def mean(self, data):
		return data.mean(axis=0)
	def stdev(self, data):
		return data.std(axis=0)
	def train(self):
		num = 1
		for x in self.classes_train:
			mean = self.mean(self.classes_train[x])
			cap = lambda x : max(x, .01)
			vfunc = np.vectorize(cap)
			stdev = vfunc(self.stdev(self.classes_train[x]))
			if x not in self.statistics:
				self.statistics[x] = (mean,stdev, self.classes_train[x].shape[0] / self.count)
			for y in range(mean.shape[0]):
				print('Class {}, attribute {}, mean = {:.2f}, std = {:.2f} '.format(int(x),num,mean[y],stdev[y]))
				num += 1
			num = 1
		#self.predict(self.classes_train[8][0,:], 8)

	def print(self):
		for x in self.classes_train_test:
			print(x, self.classes_train_test[x].shape)
			print(self.classes_train_test[x])
	def gauss(self, x, mean, stdev):
		first = 1 / ( stdev * np.sqrt(2 * np.pi))
		second = np.exp(-(np.power(x-mean, 2)) / (2 * np.power(stdev, 2)))
		return first * second
	def predict(self, x):
		oh = []
		hm = {}
		p_x = 0
		for k in self.classes_train:
			mean, stdev, prior = self.statistics[k]
			p_x_c = 1
			for i, j in enumerate(x):
				p_x_c *= self.gauss(j, mean[i], stdev[i])
			if k not in hm:
				hm[k] = (p_x_c, prior)
			else:
				hm[k] = (p_x_c, prior)
			p_x += p_x_c * prior
		for label in self.statistics:
			p_x_c2, prior2 = hm[label]
			numerator = p_x_c2 * prior2
			oh.append(numerator/ p_x)
		return oh
	def run_predictions(self):
		count = 1
		for x in self.test_data:
			class_label = int(x[-1])
			oh = self.predict(x[:-1])
			print(oh)
			# predicted = int(self.get_one_hot(oh))
			# accuracy = 0 if predicted != class_label else 1
			# print('ID={: 5d}, predicted={:3d}, probability = {:.4f}, true={: 3d}, accuracy={: 4.2f}'.format(count, int(predicted), oh[predicted], int(class_label), accuracy))
			# count += 1
	def get_one_hot(self, x):
		return np.argmax(x)










		

def main():
	if len(sys.argv) < 3:
		print('Usage: [path to training file] [path to test file]')
	classifier = NaiveBayes(sys.argv[1], sys.argv[2])
	classifier.run_predictions()
	#classifier.print()

if __name__ == '__main__':
	main()
