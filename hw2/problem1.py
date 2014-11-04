from cvxopt import matrix, solvers
import math

################################################################################
def dot(x1, x2):
  if len(x1) != len(x2):
    assert False
  return sum([x1[i] * x2[i] for i in range(len(x1))])

def scalar_mult(x, a):
    return [elem * a for elem in x]

def component_wise_add(x1, x2):
    if len(x1) != len(x2):
        assert False
    return [x1[i] + x2[i] for i in range(len(x1))]

def component_wise_subtract(x1, x2):
  if len(x1) != len(x2):
    assert False
  return [x1[i] - x2[i] for i in range(len(x1))]
################################################################################

def dot_kernel(X):
  return lambda xi, xj: dot(xi, xj)

def gaussian_kernel(X, beta):
  return lambda xi, xj: math.e ** (-1 * beta * dot(component_wise_subtract(xi, xj), component_wise_subtract(xi, xj)))
    

def build_matrices(kernel, X, Y, C):
  n = len(Y)
  q = [-1]*n
  G = [[0]*(2 * i) + [1, -1] + [0] * (2*(n - i - 1)) for i in range(n)]
  h = [C, 0]*n
  A = [[y] for y in Y]
  b = [0]
  P = [[0] * n for i in range(n)]
  for i in range(n):
    for j in range(n):
        P[i][j] = Y[i] * Y[j] * (kernel(X[i], X[j]))
  return P, q, G, h, A, b


def solve_qp(kernel, X, Y, C):
    P, q, G, h, A, b = build_matrices(kernel, X, Y, C)
    print matrix(P).size
    print matrix(q).size
    print matrix(G).size
    print matrix(h).size
    print matrix(A).size
    print matrix(b).size
    sol = solvers.qp(matrix(P, tc = 'd'), matrix(q, tc = 'd'), matrix(G, tc = 'd'), matrix(h, tc = 'd'), matrix(A, tc = 'd'), matrix(b, tc = 'd')) 
    return sol['x']


#X = [[1, 2], [2, 2], [0, 0], [-2, 3]]
#Y = [1, 1, -1, -1]
#alpha = solve_qp(X, Y, 1)

def get_weights(X, Y, alpha):
    w = [0] * len(X[0])
    for i in range(len(X)):
        term = scalar_mult(X[i], alpha[i] * Y[i])
        w = component_wise_add(w, term)
    return [get_w0(X, Y, alpha)] + w

THRESHOLD = 0.0000001

def get_w0(X, Y, alpha):
    s = 0
    count = 0
    for j in range(len(X)):
        if alpha[j] > THRESHOLD:
            inner = 0
            for i in range(len(X)):
                inner += alpha[i] * Y[i] * dot(X[i], X[j])
            s += Y[j] - inner
            count += 1
    return s/count
        
#print get_weights(X, Y, alpha)



