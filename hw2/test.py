import pylab as pl
import numpy

def anotherPlot(X, Y):
    fig1 = pl.figure(1)
    rect = fig1.patch
    rect.set_facecolor('white')
    ax = fig1.add_subplot(1,1,1)
    ax.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)
    ax.plot([-2, -1, 0, 1, 2], [4.75, 3.25, 1.75, 0.25, -1.25], 'black', linewidth = 2)
    pl.xlim([-3, 3])
    pl.ylim([-1, 4])
    pl.show()

def semiPlot(X, Yd):
  for Y in Yd:
    pl.semilogx(X, Yd[Y], label = Y)
  pl.legend()
  pl.show()

def testPlot(X, Y, w):
    print "W", w
    fig1 = pl.figure(1)
    fig1.set_figheight(11)
    fig1.set_figwidth(8.5)

    rect = fig1.patch
    rect.set_facecolor('white') 

    ax = fig1.add_subplot(1,1,1)


    ax.plot(X.T.tolist()[0],Y.T.tolist()[0], 'o', mfc="none", mec='b', mew = '2')
    # produce a plot of the values of the function 
    pts = np.array([p for p in pl.linspace(min(X), max(X), 100)])
    Yp = pl.dot(w.T, designMatrix(pts, (len(w) - 1)).T)
    ax.plot(pts, Yp, '#00ff00', linewidth = 2)
    pl.show()

X = numpy.array([[1, 2], [2, 2], [0, 0], [-2, 3]])
Y = numpy.array([[1], [1], [-1], [-1]])
#anotherPlot(X, Y)

X = numpy.array([[0.01], [0.1], [1], [10], [100]])
Y1 = numpy.array([[1.649], [1.054], [0.572], [0.435], [0.435]])
Y2 = numpy.array([[1.875], [1.318], [1.133], [1.130], [1.130]])
semiPlot(X, {"stdev1": Y1, "stdev2":Y2})

Y1 = numpy.array([[52], [16], [5], [4], [3]])
Y2 = numpy.array([[125], [96], [91], [90], [90]])
semiPlot(X, {"stdev1": Y1, "stdev2":Y2})
