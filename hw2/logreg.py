import numpy as np
import math
from scipy.optimize import fmin_bfgs
import string
import time

import matplotlib.pyplot as pl

EPSILON = 0.001

# complexity proportional to n^2
def nllUsingAlpha(alpha, xxt, y, lamda):
	sumOfLogs = 0

	n = np.size(xxt, 0)
	
#	print "Alpha:", alpha[0]
	for i in range(n):
		
		if -y[i]*np.dot(xxt[i], alpha) > 20: 
			# in this case the 1 is negligible
			sumOfLogs += -y[i]*np.dot(xxt[i], alpha)
 
		else:
			sumOfLogs += math.log(1 + math.exp(-y[i]*np.dot(xxt[i], alpha)))
	
	return sumOfLogs + lamda * sum([math.sqrt(i**2 + EPSILON) for i in alpha])

# complexity proportional to d*n
def nllUsingW(w, x, y, lamda):
	sumOfLogs = 0

	n = np.size(x, 0)
	
	print "W:", w
	for i in range(n):
		
		if -y[i]*(np.dot(x[i], w)) > 20:
			sumOfLogs += -y[i]*np.dot(x[i], w)
	
		else:
			sumOfLogs += math.log(1 + math.exp(-y[i]*np.dot(x[i], w)))

	return sumOfLogs + lamda * sum([math.sqrt(i**2 + EPSILON) for i in w])

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

	# lamda is coefficient of regularization penalty
	def findOptimalAlpha(self, lamda):
		return fmin_bfgs(nllUsingAlpha, np.array([[1./self.n]]*self.n), args=(self.xxt, self.y, lamda), norm=-float("Inf"), retall=False)

	def findOptimalW(self, lamda):
		return fmin_bfgs(nllUsingW, np.array([[1./self.d]]*self.d), args=(self.x, self.y, lamda), norm=-float("Inf"), retall=False)

#klr = KLR(np.array([[1,2,3],[1,4,5]]), np.array([[1],[-1]]))

#alphaStar = klr.findOptimalAlpha(1)
#wStar = klr.findOptimalW()

#print np.dot(klr.x.transpose(), alphaStar), wStar
#print [np.dot(klr.x.transpose(), alphaStar)[i] / wStar[i] for i in range(klr.d)]

data2dFile = open("newData/data_stdev2_test.csv", "r")

xList = []
yList = []

for line in data2dFile.readlines():
	miniList = []

	listOfNumsGoingIn = string.split(line)
	numOfNumsGoingIn = len(listOfNumsGoingIn)
	for numString in listOfNumsGoingIn[:numOfNumsGoingIn-1]:
		miniList.append(float(numString))
	
	xList.append(miniList)
	yList.append([float(listOfNumsGoingIn[numOfNumsGoingIn-1])])

klr = KLR(np.array(xList), np.array(yList))

t = time.time()
alphaStar = klr.findOptimalAlpha(1.0)
print time.time() - t
print alphaStar

wStar = klr.x.transpose().dot(alphaStar)
#wStar = [-0.02470123, -0.02373436]


print "W*", wStar

for dataPointIndex in range(klr.n):
	if yList[dataPointIndex][0] == 1.0:
#		print abs(alphaStar[dataPointIndex])
		if abs(alphaStar[dataPointIndex]) > 0.002:
			pl.plot(xList[dataPointIndex][0], xList[dataPointIndex][1], "bo")
		else:
			pl.plot(xList[dataPointIndex][0], xList[dataPointIndex][1], "bx")
	else:
		if abs(alphaStar[dataPointIndex]) > 0.002:
			pl.plot(xList[dataPointIndex][0], xList[dataPointIndex][1], "ro")
		else:
			pl.plot(xList[dataPointIndex][0], xList[dataPointIndex][1], "rx")

pl.plot([-10, 10], [-10*-wStar[1]/wStar[0], 10*-wStar[1]/wStar[0]], "k-")
	
pl.savefig("stdev2_test_plot_lambda_sup_vectors.png")		
pl.show()



#print klr.x.dot(klr.x.transpose())

#alphaStar = np.invert(klr.x.dot(klr.x.transpose())).dot(klr.x).dot(wStar) 

#print alphaStar

