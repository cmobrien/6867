import csv
import random

def split():
  f_train = open("kaggle_train.csv", 'wb')
  f_validate = open("kaggle_validate.csv", 'wb')
  f_test = open("kaggle_test.csv", 'wb')

  f_data = open("kaggle_data.csv", 'r')
  lines = csv.reader(f_data, delimiter = ",")
 
  writer_train = csv.writer(f_train, lineterminator="\n")
  writer_validate = csv.writer(f_validate, lineterminator="\n")
  writer_test = csv.writer(f_test, lineterminator="\n")

  train = []
  validate = []
  test = []

  i = 0
  for line in lines:
    if i > 1800:
      break
    if i > 0:
      choice = random.randrange(0, 3)
      if choice == 0:
        train.append(line)
      elif choice == 1:
        validate.append(line)
      else:
        test.append(line)
    i += 1

  writer_train.writerows(train)
  writer_validate.writerows(validate)
  writer_test.writerows(test)

  f_train.close()
  f_validate.close()
  f_test.close()
  f_data.close()

split()
