# param

import numpy as np

class QuadParams:
	def __init__(self):
		self.m = 1.5 # mass in kg
		self.g = 9.81 # gravity in m/s^2
		self.delta = 1

		#inertia matrix

		self.J = np.array([
			[0.02, 0.02, 0.01],
			[0.03, 0.04, 0.03],
			[0.04, 0.09, 0.04]
			
		])
