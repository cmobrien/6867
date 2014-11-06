import csv

def load_data(filename):
  f = open(filename, "r")
  lines = csv.reader(f, delimiter = ",")
  X = []
  Y = []
  for line in lines:
    X.append([int(e) for e in line[1:-1]])
    Y.append(int(line[-1]))
  f.close()
  return X, Y

def load_train():
  return load_data("data/kaggle_train_small.csv")

def load_validate():
  return load_data("data/kaggle_validate_small.csv")

def load_test():
  return load_data("data/kaggle_test_small.csv")
