import numpy as np
from math import sqrt
from copy import deepcopy
from scipy.optimize import NonlinearConstraint, minimize

class DominantRegion(object):
	def __init__(self, r, a, xi, xds):
		self.r = r
		self.a = a
		self.xi = xi
		self.xds = xds

	def level(self, x):
		for i, xd in enumerate(self.xds):
			if i == 0:
				inDR = self.a*np.linalg.norm(x-self.xi) - (np.linalg.norm(x-xd) - self.r)
			else:
				inDR = max(inDR, self.a*np.linalg.norm(x-self.xi) - (np.linalg.norm(x-xd) - self.r))
		return inDR					


class LineTarget(object):
	"""docstring for LineTarget"""
	def __init__(self, y0=0):
		self.y0 = y0
		self.type = 'line'

	def level(self, x):
		# print(x)
		return x[1] - self.y0

	def deepest_point_in_dr(self, dr, target=None):
		if target is not None:
			def obj(x):
				return max(dr.level(x), -target.level(x))
		else:
			def obj(x):
				return dr.level(x)
		in_dr = NonlinearConstraint(obj, -np.inf, 0)
		sol = minimize(self.level, dr.xi, constraints=(in_dr,))
		return sol.x

class CircleTarget():
	def __init__(self, R):
		self.R = R
		self.type = 'circle'

	def level(self, x):
		return sqrt(x[0]**2 + x[1]**2) - self.R	

	def deepest_point_in_dr(self, dr):
		in_dr = NonlinearConstraint(dr.level, -np.inf, 0)
		sol = minimize(self.level, dr.xi, constraints=(in_dr,))
		return sol.x