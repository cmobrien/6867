import numpy as np
import math
from scipy.optimize import fmin_bfgs

# complexity proportional to n^2
def nllUsingAlpha(alpha, xxt, y):
	sumOfLogs = 0

	n = np.size(xxt, 0)

	print "Alpha:", alpha
	for i in range(n):
		
		if -y[i]*np.dot(xxt[i], alpha) > 20: 
			# in this case the 1 is negligible
			sumOfLogs += -y[i]*np.dot(xxt[i], alpha)
 
		else:
			sumOfLogs += math.log(1 + math.exp(-y[i]*np.dot(xxt[i], alpha)))
	
	return sumOfLogs

# complexity proportional to d*n
def nllUsingW(w, x, y):
	sumOfLogs = 0

	n = np.size(x, 0)
	
	print "W:", w
	for i in range(n):
		
		if -y[i]*(np.dot(x[i], w)) > 20:
			sumOfLogs += -y[i]*np.dot(x[i], w)
	
		else:
			sumOfLogs += math.log(1 + math.exp(-y[i]*np.dot(x[i], w)))

	return sumOfLogs

class KLR:
	def __init__(self, x, y):
		self.x = x		
		self.xxt = x.dot(x.transpose())	
		self.y = y		

		# number of data points
		self.n = np.size(self.x, 0)

		# dimension of feature vectors 
		self.d = np.size(self.x, 1)

	# we expect alpha to be an n-dimensional vector

	def findOptimalAlpha(self):
		return fmin_bfgs(nllUsingAlpha, np.array([[1./self.n]]*self.n), args=(self.xxt, self.y), norm=-float("Inf"))

	def findOptimalW(self):
		return fmin_bfgs(nllUsingW, np.array([[1./self.d]]*self.d), args=(self.x, self.y), norm=-float("Inf"))

klr = KLR(np.array([[1,2,3],[1,4,5]]), np.array([[1],[-1]]))

alphaStar = klr.findOptimalAlpha()
wStar = klr.findOptimalW()

print np.dot(klr.x.transpose(), alphaStar), wStar
print [np.dot(klr.x.transpose(), alphaStar)[i] / wStar[i] for i in range(klr.d)]
