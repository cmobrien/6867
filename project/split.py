import csv
import random

def split():
  f_train = open("train.csv", 'wb')
  f_validate = open("validate.csv", 'wb')
  #f_test = open("test.csv", 'wb')

  f_data = open("DATA.csv", 'r')
  reader = csv.reader(f_data, delimiter = ",")
 
  writer_train = csv.writer(f_train, lineterminator="\n")
  writer_validate = csv.writer(f_validate, lineterminator="\n")
  #writer_test = csv.writer(f_test, lineterminator="\n")

  train = []
  validate = []
  #test = []

  size = 469

  lines = []
  d = {}
  i = 0
  for line in reader:
    lines.append(line)
    d[i] = True
    i += 1

  train_id = random.sample(d, size)
  for i in train_id:
    train.append(lines[i])
    del d[i]

  validate_id = random.sample(d, size)
  for i in validate_id:
    validate.append(lines[i])
    del d[i]
  
  #test_id = random.sample(d, size - 1) 
  #for i in test_id:
  #  test.append(lines[i])
  writer_train.writerows(train)
  writer_validate.writerows(validate)
  #writer_test.writerows(test)

  f_train.close()
  f_validate.close()
  #f_test.close()
  f_data.close()

split()
