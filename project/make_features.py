import csv
import numpy as np

def get_simple(n, filename):
  with open(filename, 'r') as csvfile:
    r = csv.reader(csvfile, delimiter = ",")
    rows = [row for row in r]

    if n == 0:
      assignments = []
    elif n == 1:
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

    X = []
    Y = []
    for row in rows:
      x = [float(row[i]) for i in assignments]
      X.append(x)
      Y.append([float(row[-2]), row[-1]])
    return X, Y



def get_data(n, filename):
  with open(filename, 'r') as csvfile:
    r = csv.reader(csvfile, delimiter = ",")
    rows = [row for row in r]

    if n == 0:
      assignments = []
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
      if n >= 1:
        final.append(pset_zeros)
      if n >= 3:
        final.append(quiz_zeros)
      final += row[18:]
      grades.append(final)

    X = []
    Y = []
    if n == 0:
      offset = -1
    elif n <= 2:
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
        if i <= n + offset:
          x.append(float(row[i]))
        if i == n + 3 + offset:
          x.append(float(row[i]))
        elif i == n + 1 + offset:
          if row[i] == "fa10" or row[i] == "fa12" or row[i] == "fa13":
            x += [1.0]
          elif row[i] == "sp10" or row[i] == "sp12" or row[i] == "sp14":
            x += [0.0]
          else:
            assert False
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
      if n <= 2:
        X.append(x)
      else:
        X.append(x[:n + 1] + x[n + 2:])
      Y.append([float(row[-2]), row[-1]])
  return X, Y

def get_train_simple(n):
  return get_simple(n, "train.csv")

def get_validate_simple(n):
  return get_simple(n, "validate.csv")

def get_test_simple(n):
  return get_simple(n, "validate.csv")

def get_train(n):
  return get_data(n, "train.csv")

def get_validate(n):
  return get_data(n, "validate.csv")

def get_test(n):
  return get_data(n, "test.csv")


def write_data(n):
  f = open(str(n) + ".csv", 'wb')
  writer = csv.writer(f, lineterminator="\n")
  X, Y = get_data(n, "DATA.csv")
  rows = []
  for i in range(len(X)):
    if Y[i][1] == "A":
      val = 3
    elif Y[i][1] == "B":
      val = 2
    elif Y[i][1] == "C":
      val = 1
    else:
      return False
    rows.append(X[i] + [val])
  writer.writerows(rows)


#for i in range(10):
# write_data(i)
