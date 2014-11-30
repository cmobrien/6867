import csv

FILENAME = "DATA.csv"

def get_data(n):
  with open(FILENAME, 'r') as csvfile:
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
    for row in grades:
      x = []
      for i in range(len(row)):
        if i < n + 2 or i == n + 4:
          x.append(float(row[i]))
        elif i == n + 2:
          if row[i] == "fa10":
            x += [1.0, 0.0, 0.0, 0.0, 0.0]
          elif row[i] == "fa12":
            x += [0.0, 1.0, 0.0, 0.0, 0.0]
          elif row[i] == "fa13":
            x += [0.0, 0.0, 1.0, 0.0, 0.0]
          elif row[i] == "sp10":
            x += [0.0, 0.0, 0.0, 1.0, 0.0]
          elif row[i] == "sp12":
            x += [0.0, 0.0, 0.0, 0.0, 1.0]
          else:
            assert False
        elif i == n + 3:
          if row[i] == "TRUE":
            x.append(1.0)
          elif row[i] == "FALSE":
            x.append(0.0)
          else:
            assert False
        elif i == n + 5:
          courses = row[i].split(";")
          v = [0.0] * 24
          for c in courses:
            v[int(c) - 1] = 1.0
          v = v[0:3] + v[4:10] + v[11:12] + v[13:18] + v[19:21] + v[23:]
          x += v
      X.append(x)
      Y.append(row[-2:])

  return X, Y
