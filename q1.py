import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# read data
def readData(path):
  data = []
  with open(path, 'r') as f:
    line = f.readline()
    while line:
      d = list(map(float, line.strip().split(',')))
      data.append(np.array(d))
      line = f.readline()
  return np.array(data)

trainData = readData('./dataset1_train.csv')
testData = readData('./dataset1_test.csv')
valData = readData('./dataset1_val.csv')
unknownData = readData('./dataset1_unknowns.csv')

# a

def MSE(y, pred_y):
  return sum((y-pred_y)**2)/len(y)

trainY = trainData[:,-1]
meanY = sum(trainY)/len(trainY)
trainMSE_a = MSE(trainY, meanY)
print('(a) MSE of train data:', trainMSE_a)

valY = valData[:,-1]
valMSE_a = MSE(valY, meanY)
print('(a) MSE of val data:', valMSE_a)

testY = testData[:,-1]
testMSE_a = MSE(testY, meanY)
print('(a) MSE of test data:', testMSE_a)

# b
print('='*20)
trainX = trainData[:,:-1]
valX = valData[:,:-1]
testX = testData[:,:-1]

lr = LinearRegression()
lr.fit(trainX, trainY)
pred_trainY = lr.predict(trainX)
pred_valY = lr.predict(valX)

print('(b) MSE of train data:', MSE(trainY, pred_trainY))
print('(b) MSE of val data:', MSE(valY, pred_valY))

for i in range(len(lr.coef_)):
  print('(b) weight of x{}: {}'.format(
    i+1, round(lr.coef_[i],3)
  ))

# c
print('='*20)

best = [None, 1000000, None, None]
for M in range(2, 11):
  pf = PolynomialFeatures(degree=M)
  trainX_poly = pf.fit_transform(trainX)
  lr = LinearRegression()
  lr.fit(trainX_poly, trainY)
  pred_valY = lr.predict(pf.transform(valX))
  print('(c) MSE of val data (M={}):'.format(M), MSE(valY, pred_valY))
  if(MSE(valY, pred_valY)<best[1]): best = [M, MSE(valY, pred_valY), pf, lr]

print('(c) the best M:', best[0])
for i in range(len(best[3].coef_)):
  print('(c) {} weight of {}: {}'.format(
    i,
    best[2].get_feature_names()[i].replace('x1', 'x2').replace('x0', 'x1'),
    round(best[3].coef_[i],3)
  ))
# d
trainData_plot = trainData[trainData[:,1]>=-0.1]
trainData_plot = trainData_plot[trainData_plot[:,1]<=0.1]
fig = plt.figure()
plt.plot(trainData_plot[:,0], trainData_plot[:,-1], '*')
x = np.linspace(-1, 1, 100)
y = x*-1.365
plt.plot(x, y)
plt.title('x1 pic of (b)')
plt.show()

fig = plt.figure()
plt.plot(trainData_plot[:,0], trainData_plot[:,-1], '*')
x = np.linspace(-1, 1, 100)
y = best[3].intercept_+x*best[3].coef_[1]+x**2*best[3].coef_[3]+x**3*best[3].coef_[6]+x**4*best[3].coef_[10]+x**5*best[3].coef_[15]
plt.plot(x, y)
plt.title('x1 pic of (c)')
plt.show()

trainData_plot = trainData[trainData[:,0]>=-0.1]
trainData_plot = trainData_plot[trainData_plot[:,0]<=0.1]
fig = plt.figure()
plt.plot(trainData_plot[:,1], trainData_plot[:,-1], '*')
x = np.linspace(-1, 1, 100)
y = x*-0.974
plt.plot(x, y)
plt.title('x2 pic of (b)')
plt.show()

fig = plt.figure()
plt.plot(trainData_plot[:,1], trainData_plot[:,-1], '*')
x = np.linspace(-1, 1, 100)
y = best[3].intercept_+x*best[3].coef_[2]+x**2*best[3].coef_[5]+x**3*best[3].coef_[9]+x**4*best[3].coef_[14]+x**5*best[3].coef_[20]
plt.plot(x, y)
plt.title('x2 pic of (c)')
plt.show()

# e
print('='*20)
testX_poly = best[2].transform(testX)
y_pred = best[3].predict(testX_poly)
print('(e) MSE of test data (M={}):'.format(best[0]), MSE(testY, y_pred))

unknownX_poly = best[2].transform(unknownData)
y_pred = best[3].predict(unknownX_poly)
with open('./(e)unknownData_pred.csv', 'w') as f:
  for i in y_pred:
    f.write(str(i))
    f.write('\n')


# f
print('='*20)

M = 1
a = 0
pf = PolynomialFeatures(degree=M)
trainX_poly = pf.fit_transform(trainX)
r = Ridge(alpha=a)
r.fit(trainX_poly, trainY)
pred_valY = r.predict(pf.transform(valX))
print('(f) MSE(M=1;a=0) is {}'.format(MSE(valY, pred_valY)))

best = [None, None, 1000000, None, None]
for M in range(2, 8):
  for a in [0,0.001,0.01,0.1,1,10,100]:
    pf = PolynomialFeatures(degree=M)
    trainX_poly = pf.fit_transform(trainX)
    r = Ridge(alpha=a)
    r.fit(trainX_poly, trainY)
    pred_valY = r.predict(pf.transform(valX))
    if(MSE(valY, pred_valY)<best[2]): best = [M, a, MSE(valY, pred_valY), pf, r]
print('(f) best M is {}'.format(best[0]))
print('(f) best alpha is {}'.format(best[1]))
print('(f) best MSE is {}'.format(best[2]))
for i in range(len(best[4].coef_)):
  print('(f) {} weight of {}: {}'.format(
    i,
    best[3].get_feature_names()[i].replace('x1', 'x2').replace('x0', 'x1'),
    round(best[4].coef_[i],3)
  ))

fig = plt.figure()
for M in range(2, 8):
  mse = []
  for a in [0,0.001,0.01,0.1,1,10,100]:
    pf = PolynomialFeatures(degree=M)
    trainX_poly = pf.fit_transform(trainX)
    r = Ridge(alpha=a)
    r.fit(trainX_poly, trainY)
    pred_valY = r.predict(pf.transform(valX))
    mse.append(MSE(valY, pred_valY))
  plt.plot(list(range(7)), mse, label="M={}".format(M))

plt.xticks(list(range(7)), [0,0.001,0.01,0.1,1,10,100])
plt.title('(f) validation-set-MSE vs. alpha')
plt.legend()
plt.show()


# h
print('='*20)
testX_poly = best[3].transform(testX)
y_pred = best[4].predict(testX_poly)
print('(h) MSE of test data (M={};a={}):'.format(best[0], best[1]), MSE(testY, y_pred))

unknownX_poly = best[3].transform(unknownData)
y_pred = best[4].predict(unknownX_poly)
with open('./(h)unknownData_pred.csv', 'w') as f:
  for i in y_pred:
    f.write(str(i))
    f.write('\n')

