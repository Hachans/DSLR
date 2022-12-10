import numpy as np
import json
import sys


class LogisticRegression():
	"""Params:
	lrate: float, Learning rate between 0 and 1, default 0.1
	epochs: int, amount of iterations over dataset, default 50
	"""
	def __init__(self, lrate=0.1, epochs=50, initial_weight=None, multi_class=None):
		self.lrate = lrate
		self.epochs = epochs
		self._w = initial_weight
		self._classes = multi_class
		self._errors = []
		self._cost = []

	def sigmoid(self, X):
		return 1 / (1 + np.exp(-(self._w.dot(X.T))))

	def fit(self, X, y):
		self._classes = np.unique(y).tolist()
		newX = np.insert(X, 0, 1, axis=1)
		m = newX.shape[0]

		self._w = np.zeros(newX.shape[1] * len(self._classes))
		self._w = self._w.reshape(len(self._classes), newX.shape[1])

		y_vec = np.zeros((len(y), len(self._classes)))
		for i in range(len(y)):
			y_vec[i, self._classes.index(y[i])] = 1

		for _ in range(self.epochs):
			predictions = self.sigmoid(newX).T

			lhs = y_vec.T.dot(np.log(predictions))
			rhs = (1 - y_vec).T.dot(np.log(1 - predictions))

			cost = (-1 / m) * sum(lhs + rhs)
			self._cost.append(cost)
			self._errors.append(sum(y != self.predict(X)))
			self._w = self._w - (self.lrate * (1 / m) * (predictions - y_vec).T.dot(newX))
		return self

	def predict(self, X):
		X = np.insert(X, 0, 1, axis=1)
		predictions = self.sigmoid(X).T
		return [self._classes[x] for x in predictions.argmax(1)]
	
	def save_values(self):
		try:
			with open("data/values.json", "r") as f:
				file_data = json.load(f)
			file_data["weights"] = self._w.tolist()
			file_data["classes"] = self._classes
			with open("data/values.json", "w") as f:
				json.dump(file_data, f)
		except:
			print("Something went wrong with opening/writing to values.json", sys.stderr)
			sys.exit(-1)

	def load_values(self, path):
		try:
			with open(path, "r") as f:
				file_data = json.load(f)
			self._w = np.array(file_data['weights'])
			self._classes = file_data['classes']
		except:
			print("Something went wrong with opening/writing to values.json", sys.stderr)
			sys.exit(-1)

