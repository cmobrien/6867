

def dot(x1, x2):
  if len(x1) != len(x2):
    assert False
  return sum([x1[i] * x2[i] for i in range(len(x1))])

def build_matrices(X, Y, C):
  n = len(X)
  p = [-1]*n
  G = [[0]*(2 * i) + [1, -1] + [0] * (2*(n - i - 1)) for i in range(n)]
  h = [C, 0]*n
  A = Y
  b = [0]
  Q = [[0] * n for i in range(n)]
  for i in range(n):
    for j in range(n):
      if i == j:
        Q[i][j] = 0.5 * Y[i] ** 2 * (dot(X[i], X[i]))
      else:
        Q[i][j] = 0.25 * Y[i] * Y[j] * (dot(X[i], X[j]))
  print Q
  return Q, p, G, h, A, b

build_matrices([[1], [2], [3], [4], [5]], [1, 2, 3, 4, 5], 1)


