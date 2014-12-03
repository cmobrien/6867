from make_features import *
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

def dot(x1, x2):
  assert len(x1) == len(x2)
  return sum([x1[i] * x2[i] for i in range(len(x1))])

def predict(x, y, w):
  return w[0] + dot(x, w[1:])

def calculate_error(guess, actual):
  error = 0
  mis = []
  assert len(guess) == len(actual)
  for i in range(len(guess)):
    if guess[i] != actual[i]:
      error += 1
      mis.append(i)
  return error, mis

L0 = \
    """f$FALL       0.08592    0.14317  0.6001
    f$MALE       0.23481    0.13099  1.7926
    f$YEAR      -0.45878    0.08672 -5.2904
    f$COURSE_6   0.28924    0.20565  1.4064
    f$COURSE_8   1.18389    0.38615  3.0659
    f$COURSE_18  0.89343    0.21764  4.1051"""
L0mA = -0.0113
L0mB = -2.0465

L1 = \
    """f$PSET1       0.53324    0.09234  5.7746
    f$PSET_ZEROS -1.12319    0.19515 -5.7555
    f$FALL        0.07666    0.14679  0.5222
    f$MALE        0.33422    0.13431  2.4884
    f$YEAR       -0.37682    0.08930 -4.2196
    f$COURSE_6    0.18884    0.20976  0.9002
    f$COURSE_8    1.05503    0.38381  2.7489
    f$COURSE_18   0.81863    0.22121  3.7006"""
L1mA = 0.1069
L1mB = -2.1423

L2 = \
    """f$PSET1       0.40453    0.09137  4.4272
    f$PSET2       0.98918    0.10896  9.0784
    f$PSET_ZEROS -0.41822    0.22087 -1.8935
    f$FALL       -0.02878    0.15112 -0.1904
    f$MALE        0.39924    0.13757  2.9020
    f$YEAR       -0.29953    0.09129 -3.2813
    f$COURSE_6    0.18693    0.21617  0.8647
    f$COURSE_8    1.06194    0.39395  2.6956
    f$COURSE_18   0.88137    0.22809  3.8641"""
L2mA = 0.4632
L2mB = -1.9878

L3 = \
    """f$PSET1       0.3079    0.11168  2.7568
    f$PSET2       0.8113    0.12332  6.5788
    f$QUIZ1       0.8647    0.04854 17.8145
    f$PSET_ZEROS -0.7913    0.26474 -2.9890
    f$FALL       -0.1098    0.18312 -0.5996
    f$MALE        0.0609    0.16433  0.3706
    f$YEAR       -0.1697    0.11363 -1.4935
    f$COURSE_6    0.4355    0.25700  1.6944
    f$COURSE_8    0.4883    0.45515  1.0727
    f$COURSE_18   0.6462    0.27992  2.3086"""
L3mA = 0.8927
L3mB = -3.0518

L4 = \
    """f$PSET1       0.21013    0.11530  1.8225
    f$PSET2       0.53777    0.12679  4.2414
    f$PSET3       0.93938    0.13346  7.0386
    f$QUIZ1       0.89358    0.05078 17.5982
    f$PSET_ZEROS -0.71865    0.26675 -2.6941
    f$FALL       -0.10972    0.18715 -0.5863
    f$MALE        0.07824    0.16757  0.4669
    f$YEAR       -0.17434    0.11484 -1.5180
    f$COURSE_6    0.29583    0.26415  1.1199
    f$COURSE_8    0.51951    0.46666  1.1132
    f$COURSE_18   0.51004    0.28763  1.7733"""
L4mA = 0.8591
L4mB = -3.3195

L5 = \
    """f$PSET1       0.1787    0.11343  1.5757
    f$PSET2       0.4118    0.12716  3.2389
    f$PSET3       0.7965    0.13631  5.8430
    f$PSET4       0.6558    0.11262  5.8230
    f$QUIZ1       0.9104    0.05238 17.3829
    f$PSET_ZEROS -0.2864    0.27463 -1.0430
    f$FALL       -0.0984    0.18917 -0.5202
    f$MALE        0.1029    0.16985  0.6060
    f$YEAR       -0.1492    0.11535 -1.2937
    f$COURSE_6    0.2814    0.26757  1.0518
    f$COURSE_8    0.7183    0.47268  1.5196
    f$COURSE_18   0.4955    0.29122  1.7016"""
L5mA = 1.0204
L5mB = -3.3517

L6 = \
    """f$PSET1       0.282985    0.12118  2.33528
    f$PSET2       0.440578    0.14337  3.07299
    f$PSET3       0.943822    0.15247  6.19041
    f$PSET4       0.684782    0.12866  5.32262
    f$QUIZ1       0.809763    0.05945 13.62041
    f$QUIZ2       0.681239    0.05585 12.19770
    f$PSET_ZEROS -0.272780    0.32508 -0.83912
    f$FALL       -0.334024    0.21501 -1.55351
    f$MALE       -0.003224    0.18945 -0.01702
    f$YEAR        0.028470    0.13329  0.21359
    f$COURSE_6    0.359313    0.30393  1.18222
    f$COURSE_8    0.923939    0.54125  1.70703
    f$COURSE_18   0.241802    0.32376  0.74686"""
L6mA = 1.7175
L6mB = -4.0557

L7 = \
    """f$PSET1       0.29973    0.12557  2.3869
    f$PSET2       0.43271    0.14621  2.9596
    f$PSET3       0.82267    0.15713  5.2356
    f$PSET4       0.55956    0.13229  4.2299
    f$PSET5       0.62605    0.12419  5.0409
    f$QUIZ1       0.83396    0.06113 13.6419
    f$QUIZ2       0.68345    0.05691 12.0095
    f$PSET_ZEROS  0.24712    0.35366  0.6987
    f$FALL       -0.34664    0.22026 -1.5738
    f$MALE        0.06096    0.19372  0.3147
    f$YEAR        0.02646    0.13590  0.1947
    f$COURSE_6    0.31102    0.30975  1.0041
    f$COURSE_8    0.96414    0.53990  1.7858
    f$COURSE_18   0.24586    0.33140  0.7419"""
L7mA = 1.8196
L7mB = -4.1334

L8 = \
    """f$PSET1       0.29242    0.13063  2.2386
    f$PSET2       0.41845    0.14653  2.8558
    f$PSET3       0.68394    0.15879  4.3073
    f$PSET4       0.51893    0.13578  3.8218
    f$PSET5       0.57325    0.12609  4.5464
    f$PSET6       0.51353    0.11705  4.3871
    f$QUIZ1       0.85652    0.06249 13.7068
    f$QUIZ2       0.67379    0.05746 11.7257
    f$PSET_ZEROS  0.73528    0.36849  1.9954
    f$FALL       -0.36263    0.22255 -1.6294
    f$MALE        0.12556    0.19651  0.6389
    f$YEAR        0.03695    0.13742  0.2689
    f$COURSE_6    0.28377    0.31238  0.9084
    f$COURSE_8    0.89802    0.53486  1.6790
    f$COURSE_18   0.21869    0.33313  0.6565"""
L8mA = 1.9433
L8mB = -4.1717

L9 = \
    """f$PSET1       0.57975    0.18000  3.2208
    f$PSET2       0.78950    0.19063  4.1415
    f$PSET3       0.83887    0.19969  4.2009
    f$PSET4       0.53724    0.17801  3.0181
    f$PSET5       1.11951    0.17382  6.4408
    f$PSET6       0.54522    0.15175  3.5929
    f$QUIZ1       0.89250    0.08695 10.2648
    f$QUIZ2       0.62293    0.07981  7.8047
    f$FINAL       0.93039    0.07702 12.0796
    f$PSET_ZEROS  0.79683    0.44473  1.7917
    f$FALL       -0.52973    0.28840 -1.8368
    f$MALE        0.06661    0.25185  0.2645
    f$YEAR        0.23769    0.17101  1.3899
    f$COURSE_6    0.73785    0.41295  1.7868
    f$COURSE_8   -0.20992    0.69058 -0.3040
    f$COURSE_18   0.19777    0.43351  0.4562"""
L9mA = 3.7016
L9mB = -6.7252

P0 = \
    """f$FALL       0.05095    0.08540  0.5966
    f$MALE       0.13556    0.07856  1.7255
    f$YEAR      -0.26863    0.04982 -5.3926
    f$COURSE_6   0.17027    0.12047  1.4133
    f$COURSE_8   0.71309    0.22190  3.2136
    f$COURSE_18  0.52662    0.12820  4.1077"""
P0mA = 0.0024
P0mB = -1.2216

P1 = \
    """f$PSET1       0.27412    0.04897  5.5982
    f$PSET_ZEROS -0.63592    0.10951 -5.8068
    f$FALL        0.03945    0.08756  0.4506
    f$MALE        0.18844    0.08002  2.3548
    f$YEAR       -0.21336    0.05175 -4.1230
    f$COURSE_6    0.10390    0.12394  0.8383
    f$COURSE_8    0.63568    0.22602  2.8124
    f$COURSE_18   0.47734    0.13130  3.6354"""
P1mA = 0.0628
P1mB = -1.2614

P2 = \
    """f$PSET1       0.22446    0.05054  4.4415
    f$PSET2       0.54224    0.05675  9.5542
    f$PSET_ZEROS -0.19137    0.12098 -1.5818
    f$FALL       -0.01386    0.08934 -0.1551
    f$MALE        0.22103    0.08133  2.7178
    f$YEAR       -0.17460    0.05319 -3.2824
    f$COURSE_6    0.11390    0.12673  0.8988
    f$COURSE_8    0.65451    0.23023  2.8428
    f$COURSE_18   0.51636    0.13420  3.8476"""
P2mA = 0.2613
P2mB = -1.1632

P3 = \
    """f$PSET1       0.17333    0.05851   2.963
    f$PSET2       0.44890    0.06461   6.948
    f$QUIZ1       0.48896    0.02506  19.513
    f$PSET_ZEROS -0.38882    0.13885  -2.800
    f$FALL       -0.08285    0.10317  -0.803
    f$MALE        0.06508    0.09310   0.699
    f$YEAR       -0.09600    0.06272  -1.531
    f$COURSE_6    0.28010    0.14758   1.898
    f$COURSE_8    0.26157    0.25556   1.024
    f$COURSE_18   0.33670    0.15895   2.118"""
P3mA = 0.5398
P3mB = -1.6854

P4 = \
    """f$PSET1       0.12916    0.06111  2.1137
    f$PSET2       0.29832    0.06947  4.2945
    f$PSET3       0.52774    0.07486  7.0494
    f$QUIZ1       0.50187    0.02610 19.2274
    f$PSET_ZEROS -0.37061    0.14586 -2.5409
    f$FALL       -0.08189    0.10517 -0.7786
    f$MALE        0.06710    0.09473  0.7083
    f$YEAR       -0.09674    0.06407 -1.5099
    f$COURSE_6    0.19468    0.15136  1.2862
    f$COURSE_8    0.27255    0.26268  1.0376
    f$COURSE_18   0.24026    0.16273  1.4764"""
P4mA = 0.5111
P4mB = -1.8335

P5 = \
    """f$PSET1       0.11555    0.06173  1.8720
    f$PSET2       0.23879    0.07078  3.3738
    f$PSET3       0.43817    0.07750  5.6535
    f$PSET4       0.37975    0.06333  5.9966
    f$QUIZ1       0.50951    0.02690 18.9406
    f$PSET_ZEROS -0.13183    0.15460 -0.8527
    f$FALL       -0.07837    0.10693 -0.7330
    f$MALE        0.07599    0.09619  0.7901
    f$YEAR       -0.08912    0.06490 -1.3732
    f$COURSE_6    0.17994    0.15420  1.1669
    f$COURSE_8    0.37532    0.26540  1.4142
    f$COURSE_18   0.23504    0.16543  1.4208"""
P5mA = 0.5826
P5mB = -1.8685

P6 = \
    """f$PSET1       0.1558558    0.06914  2.254310
    f$PSET2       0.2260376    0.07787  2.902685
    f$PSET3       0.5328506    0.08561  6.224198
    f$PSET4       0.3905939    0.07226  5.405677
    f$QUIZ1       0.4544429    0.03167 14.350865
    f$QUIZ2       0.3815619    0.02981 12.799663
    f$PSET_ZEROS -0.1620936    0.18061 -0.897473
    f$FALL       -0.1950669    0.11921 -1.636363
    f$MALE        0.0040045    0.10645  0.037619
    f$YEAR        0.0006564    0.07338  0.008945
    f$COURSE_6    0.1929806    0.17340  1.112918
    f$COURSE_8    0.4934722    0.29542  1.670408
    f$COURSE_18   0.0975630    0.18377  0.530889"""
P6mA = 0.9061 
P6mB = -2.2918

P7 = \
    """f$PSET1       0.156444    0.07128  2.19466
    f$PSET2       0.230277    0.07928  2.90442
    f$PSET3       0.467178    0.08804  5.30665
    f$PSET4       0.319511    0.07448  4.29000
    f$PSET5       0.339776    0.06974  4.87195
    f$QUIZ1       0.465148    0.03232 14.39131
    f$QUIZ2       0.378564    0.03014 12.56141
    f$PSET_ZEROS  0.157322    0.19548  0.80481
    f$FALL       -0.201034    0.12077 -1.66466
    f$MALE        0.043140    0.10826  0.39848
    f$YEAR       -0.001068    0.07456 -0.01433
    f$COURSE_6    0.175944    0.17590  1.00026
    f$COURSE_8    0.546789    0.29838  1.83254
    f$COURSE_18   0.096435    0.18657  0.51688"""
P7mA = 0.9686
P7mB = -2.3062

P8 = \
    """f$PSET1       0.144459    0.07181  2.0117
    f$PSET2       0.236012    0.07935  2.9744
    f$PSET3       0.387015    0.08956  4.3214
    f$PSET4       0.296991    0.07600  3.9076
    f$PSET5       0.302278    0.07056  4.2842
    f$PSET6       0.280536    0.06340  4.4248
    f$QUIZ1       0.476702    0.03286 14.5086
    f$QUIZ2       0.373819    0.03049 12.2585
    f$PSET_ZEROS  0.434630    0.20306  2.1404
    f$FALL       -0.216353    0.12194 -1.7743
    f$MALE        0.078315    0.10959  0.7146
    f$YEAR        0.008575    0.07560  0.1134
    f$COURSE_6    0.175595    0.17775  0.9879
    f$COURSE_8    0.508400    0.29956  1.6971
    f$COURSE_18   0.095513    0.18814  0.5077"""
P8mA = 1.0587
P8mB = -2.3068

P9 = \
    """f$PSET1       0.36140    0.09315  3.87978
    f$PSET2       0.46298    0.10204  4.53707
    f$PSET3       0.44203    0.10684  4.13716
    f$PSET4       0.30186    0.09725  3.10404
    f$PSET5       0.61684    0.09580  6.43867
    f$PSET6       0.28419    0.07875  3.60891
    f$QUIZ1       0.48731    0.04539 10.73547
    f$QUIZ2       0.34645    0.04252  8.14768
    f$FINAL       0.50313    0.03904 12.88666
    f$PSET_ZEROS  0.46777    0.24285  1.92618
    f$FALL       -0.22860    0.15429 -1.48163
    f$MALE        0.04981    0.13920  0.35785
    f$YEAR        0.12295    0.09548  1.28771
    f$COURSE_6    0.48767    0.23183  2.10360
    f$COURSE_8   -0.00537    0.38591 -0.01392
    f$COURSE_18   0.15209    0.24391  0.62352"""
P9mA = 2.1239
P9mB = -3.5740

def get_weights(S):
  w = [0]
  f = S.split("$")
  for i in range(1, len(f)):
    elems = f[i].split(" ")
    for j in range(1, len(elems)):
      if elems[j] != '':
        w.append(float(elems[j]))
        break
  return w

a = [(P0, P0mA, P0mB),
     (P1, P1mA, P1mB),
     (P2, P2mA, P2mB),
     (P3, P3mA, P3mB),
     (P4, P4mA, P4mB),
     (P5, P5mA, P5mB),
     (P6, P6mA, P6mB),
     (P7, P7mA, P7mB),
     (P8, P8mA, P8mB),
     (P9, P9mA, P9mB)
    ]

#a = [(L0, P0mA, L0mB),
#     (L1, L1mA, L1mB),
#     (L2, L2mA, L2mB),
#     (L3, L3mA, L3mB),
#     (L4, L4mA, L4mB),
#     (L5, L5mA, L5mB),
#     (L6, L6mA, L6mB),
#     (L7, L7mA, L7mB),
#     (L8, L8mA, L8mB),
#     (L9, L9mA, L9mB)
#    ]

weights_and_cutoffs = [(get_weights(r[0]), r[1], r[2]) for r in a]

def get_dist_probit(pred, mA, mB):
  A = 1 - norm.cdf(mA - pred)
  B = norm.cdf(mA - pred) - norm.cdf(mB - pred)
  C = norm.cdf(mB - pred)
  return (A, B, C)

def logit(x):
  return 1.0 / (1 + math.pow(math.e, -1 * x))

def get_dist_logit(pred, mA, mB):
  A = 1 - logit(mA - pred)
  B = logit(mA - pred) - logit(mB - pred)
  C = logit(mB - pred)
  return (A, B, C)


def get_penalty(G, dist):
  if G == "A":
    return math.log(dist[0])
  if G == "B":
    return math.log(dist[1])
  if G == "C":
    return math.log(dist[2])

k = 0
#for w, mA, mB in weights_and_cutoffs:
#  print
#  print k
#  guess = []
#
#  X, Y = get_data(k, "test.csv") 
#  Y_letter = [y[1] for y in Y]
#  penalty = 0
#  for i in range(len(X)):
#    val = predict(X[i], Y_letter[i], w)
#    dist = get_dist_logit(val, mA, mB)
#    if val > mA:
#      guess.append("A")
#    elif val > mB:
#      guess.append("B")
#    else:
#      guess.append("C")
#    p = get_penalty(Y_letter[i], dist)
#    penalty += p
#  k += 1
#  errors, mis = calculate_error(guess, Y_letter)
#  print errors
#  print penalty



k = 3
w, mA, mB = weights_and_cutoffs[k]
X, Y = get_data(k, "test.csv") 
Y_letter = [y[1] for y in Y]
penalty = 0
guess = []
n = 100
for i in range(len(X)):
  val = predict(X[i], Y_letter[i], w)
  dist = get_dist_probit(val, mA, mB)
  print i, dist
  if val > mA:
    guess.append("A")
  elif val > mB:
    guess.append("B")
  else:
    guess.append("C")
  p = get_penalty(Y_letter[i], dist)
  penalty += p
k += 1
errors, mis = calculate_error(guess, Y_letter)
print errors
print penalty

#x_plot = []
#y_plot = []
#for THRESHOLD in [i/float(n) for i in range(n + 1)]:
#  count = 0
#  for m in mis:
#    dist = get_dist_logit(predict(X[m], Y_letter[m], w), mA, mB)
#    if dist[0] > THRESHOLD or dist[1] > THRESHOLD or dist[2] > THRESHOLD:
#      count += 1
#  print THRESHOLD, count
#  x_plot.append(THRESHOLD)
#  y_plot.append(count)

#print x_plot
#print y_plot
#plt.plot(x_plot, y_plot)
#plt.show()
