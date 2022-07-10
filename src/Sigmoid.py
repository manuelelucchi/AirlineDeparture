import numpy as np


def sigmoid(data):
	'''
	Calculates the sigmoid of the given data
	'''
	g = 1.0 / (1.0 + np.exp(-data))	
	return g