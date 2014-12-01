import csv
import numpy as np

def get_data(n, filename):
  with open(filename, 'r') as csvfile:
    r = csv.reader(csvfile, delimiter = ",")
    rows = [row for row in r]

    if n == 1:
      assignments = [0]
    elif n == 2:
      assignments = [0, 2]
    elif n == 3:
      assignments = [0, 2, 12]
    elif n == 4:
      assignments = [0, 2, 4, 12]
    elif n == 5:
      assignments = [0, 2, 4, 6, 12]
    elif n == 6:
      assignments = [0, 2, 4, 6, 12, 14]
    elif n == 7:
      assignments = [0, 2, 4, 6, 8, 12, 14]
    elif n == 8:
      assignments = [0, 2, 4, 6, 8, 10, 12, 14]
    elif n == 9:
      assignments = [0, 2, 4, 6, 8, 10, 12, 14, 16]

    grades = []
    for row in rows:
      final = []
      pset_zeros = 0
      quiz_zeros = 0
      for j in assignments:
        final.append(row[j])
        if row[j + 1] == '1':
          if j < 12:
            pset_zeros += 1
          else:
            quiz_zeros += 1
      final.append(pset_zeros)
      if j >= 12:
        final.append(quiz_zeros)
      final += row[18:]
      grades.append(final)

    X = []
    Y = []
    if n <= 2:
      offset = 0
    else:
      offset = 1
    fa10 = [[float(a) for a in r[0:n]] for r in grades if r[n + 1 + offset] == "fa10"]
    fa12 = [[float(a) for a in r[0:n]] for r in grades if r[n + 1 + offset] == "fa12"]
    fa13 = [[float(a) for a in r[0:n]] for r in grades if r[n + 1 + offset] == "fa13"]
    sp10 = [[float(a) for a in r[0:n]] for r in grades if r[n + 1 + offset] == "sp10"]
    sp12 = [[float(a) for a in r[0:n]] for r in grades if r[n + 1 + offset] == "sp12"]
    for row in grades:
      x = []
      for i in range(len(row)):
        if i < n + 1 + offset or i == n + 3 + offset:
          x.append(float(row[i]))
        elif i == n + 1 + offset:
          if row[i] == "fa10":
            #x += [1.0, 0.0, 0.0, 0.0, 0.0]
            fa10.remove(x[0:n])
            s = np.std(fa10, axis = 0).tolist()
            fa10.append(x[0:n])
          elif row[i] == "fa12":
            #x += [0.0, 1.0, 0.0, 0.0, 0.0]
            fa12.remove(x[0:n])
            s = np.std(fa12, axis = 0).tolist()
            fa12.append(x[0:n])
          elif row[i] == "fa13":
            #x += [0.0, 0.0, 1.0, 0.0, 0.0]
            fa13.remove(x[0:n])
            s = np.std(fa13, axis = 0).tolist()
            fa13.append(x[0:n])
          elif row[i] == "sp10":
            #x += [0.0, 0.0, 0.0, 1.0, 0.0]
            sp10.remove(x[0:n])
            s = np.std(sp10, axis = 0).tolist()
            sp10.append(x[0:n])
          elif row[i] == "sp12":
            #x += [0.0, 0.0, 0.0, 0.0, 1.0]
            sp12.remove(x[0:n])
            s = np.std(sp12, axis = 0).tolist()
            sp12.append(x[0:n])
          else:
            assert False
          x += s
        elif i == n + 2 + offset:
          if row[i] == "TRUE":
            x.append(1.0)
          elif row[i] == "FALSE":
            x.append(0.0)
          else:
            assert False
        elif i == n + 4 + offset:
          courses = row[i].split(";")
          v = [0.0] * 24
          for c in courses:
            v[int(c) - 1] = 1.0
          #v = v[0:3] + v[4:10] + v[11:12] + v[13:18] + v[19:21] + v[23:]
          v = [v[5]] + [v[7]] + [v[17]]
          x += v
      X.append(x)
      Y.append([float(row[-2]), row[-1]])

  return X, Y


def get_train(n):
  return get_data(n, "train.csv")

def get_validate(n):
  return get_data(n, "validate.csv")

def get_test(n):
  return get_data(n, "test.csv")
