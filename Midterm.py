#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 01:42:32 2021

@author: maoyuqiang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
np.set_printoptions(suppress=True)

# load train, validation and test datasets, and the unknows
df_train = pd.read_csv('dataset1_train.csv', sep = ',', header = None).to_numpy()
df_val = pd.read_csv('dataset1_val.csv', sep = ',', header = None).to_numpy()
df_test = pd.read_csv('dataset1_test.csv', sep = ',', header = None).to_numpy()
df_unknowns = pd.read_csv('dataset1_unknowns.csv', sep = ',', header = None).to_numpy()
"""
Part(a)

function that compute the MSE of the system

input is the full training data
 
"""

def MSE_Compute(train): 
    y = train[:,-1]
    y_mean = np.mean(train[:,-1], axis = 0) 
    res = np.mean(np.square(y - y_mean), axis = 0)
    return res
print('Part(a)\n')
# calculate the mse on the training set
print('MSE on the training set: {} \n'.format(MSE_Compute(df_train)))
# calculate the mse on the validation set
print('MSE on the validation set: {} \n'.format(MSE_Compute(df_val)))
# calculate the mse on the test set
print('MSE on the test set: {} \n'.format(MSE_Compute(df_test)))

"""
Part(b) - (e)

""" 
# (b)
lr = LinearRegression(fit_intercept=True)
lr.fit(df_train[:,:-1], df_train[:,-1])
y_pred_train = lr.predict(df_train[:,:-1])
y_pred_val = lr.predict(df_val[:,:-1])

print('Part(b)\n')
# calculate the resulting MSE on the training set. 
print("The resulting MSE on the training set: {} \n".format(
    mean_squared_error(df_train[:,-1], y_pred_train)))
# calculate the resulting MSE on the validation set.
print("the resulting MSE on the validation set: {} \n".format(
    mean_squared_error(df_val[:,-1], y_pred_val)))

# report the final(optimal) weight values.
opt_train_weights = np.concatenate((np.array([lr.intercept_]),lr.coef_))
print('''The final(optimal) weight values for MSE regression on the training data
{} \n'''.format(opt_train_weights.round(decimals=3)))

# (c)
"""
 funtion that find the optimal M(degree of polynomial)
 return the optimized degree M
"""
def model_selection_mse(df_train, df_val):
    minimum_mse_on_val = float('inf')
    degrees = [i for i in range(2,11)]
    ## iterate possible degrees
    for i, degree in enumerate(degrees):
        poly_lr = LinearRegression()
        # generate the expanded feature space
        poly = PolynomialFeatures(degree=degree, include_bias=False)     
        # train the model            
        poly_lr.fit(poly.fit_transform(df_train[:,:-1]), df_train[:,-1])
        pred = poly_lr.predict(poly.fit_transform(df_val[:,:-1]))
        # find the MSE of the current model with degree M
        curr_mse = mean_squared_error(df_val[:,-1], pred)    
        # Cross-validation of degree
        if curr_mse < minimum_mse_on_val:
            minimum_mse_on_val = curr_mse
            minimum_degree = degree
    return minimum_degree
opt_degree_M = model_selection_mse(df_train, df_val)
poly_lr = LinearRegression(fit_intercept=True)
poly = PolynomialFeatures(degree=opt_degree_M, include_bias=False)
poly_lr.fit(poly.fit_transform(df_train[:,:-1]), df_train[:,-1])    
opt_train_weights_M_degree = np.concatenate((np.array([poly_lr.intercept_]),
                              poly_lr.coef_))

print('Part(c)\n')
# report the final weight values
print("The final weight values under optimized degree M = {} is \n {}\n".format(
opt_degree_M, opt_train_weights_M_degree.round(decimals=3)))

# report the MSE on the training set and validation set. 
y_pred_train = poly_lr.predict(poly.fit_transform(df_train[:,:-1]))
y_pred_val = poly_lr.predict(poly.fit_transform(df_val[:,:-1]))
print("The resulting MSE on the training set: {} \n".format(
    mean_squared_error(df_train[:,-1], y_pred_train)))
print("the resulting MSE on the validation set: {} \n".format(
    mean_squared_error(df_val[:,-1], y_pred_val)))

# (d)
# for (b)
x1 = df_train[np.logical_and(df_train[:,1]<=0.1, df_train[:,1]>= -0.1)][:,0]
x2 = df_train[np.logical_and(df_train[:,0]<=0.1, df_train[:,0]>= -0.1)][:,1]
y1 = df_train[np.logical_and(df_train[:,1]<=0.1, df_train[:,1]>= -0.1)][:,-1]
y2 = df_train[np.logical_and(df_train[:,0]<=0.1, df_train[:,0]>= -0.1)][:,-1]

fig1, (ax1, ax2) = plt.subplots(1, 2)
fig1.suptitle('Plots for result of (b)')
ax1.scatter(x1,y1, label = '(x1,y)')
ax1.set_xlabel('x1')
ax1.set_ylabel('y', rotation = 'horizontal')
intercept = opt_train_weights[0]      
slope = opt_train_weights[1]       
x = np.linspace(-1.0,1.0)
## calculate and plot the decision boundary
ax1.plot(x, slope*x + intercept, color = 'r', label = 'y_predict')
ax1.legend(loc="upper right")
ax1.set_title('(i)')

ax2.scatter(x2, y2, label = '(x2,y)')
ax2.set_xlabel('x2')
intercept = opt_train_weights[0]      
slope = opt_train_weights[2]       
x = np.linspace(-1.0,1.0)
## calculate and plot the decision boundary
ax2.plot(x, slope*x + intercept, color = 'r', label = 'y_predict')
ax2.legend(loc="upper right")
ax2.set_title('(ii)')
plt.tight_layout() 
plt.figsize=(10, 6)

# for (c)
fig2, (ax3, ax4) = plt.subplots(1, 2)
plt.figsize=(20, 16)
fig2.suptitle('Plots for result of (c)')
ax3.scatter(x1,y1, label = '(x1,y)')
ax3.set_xlabel('x1')
ax3.set_ylabel('y', rotation = 'horizontal')
new_x1 = df_train[np.logical_and(df_train[:,1]<=0.1, df_train[:,1]>= -0.1)]
# with x2 = 0
new_x1[:,1]=0
poly = PolynomialFeatures(degree=opt_degree_M, include_bias=False)
y_pred_train = poly_lr.predict(poly.fit_transform(new_x1[:,:-1]))
x = new_x1[:,0]
y = y_pred_train
## find the coefficients for the polynomial
## plot the decision boundary
coefficients = np.polyfit(x, y, 5)
polynomial = np.poly1d(coefficients)
new_xx = np.linspace(-1.0,1.0)
new_yy = polynomial(new_xx)
ax3.plot(new_xx, new_yy,color = 'r', label = 'y_predict under M = 5')
ax3.legend(loc="upper right")
ax3.set_title('(i)')

ax4.scatter(x2, y2, label = '(x2,y)')
ax4.set_xlabel('x2')
new_x2 = df_train[np.logical_and(df_train[:,0]<=0.1, df_train[:,0]>= -0.1)]
# with x1 = 0
new_x2[:,0]=0
poly = PolynomialFeatures(degree=opt_degree_M, include_bias=False)
y_pred_train = poly_lr.predict(poly.fit_transform(new_x2[:,:-1]))
x = new_x2[:,1]
y = y_pred_train
## find the coefficients for the polynomial
## plot the decision boundary
coefficients = np.polyfit(x, y, 5)
polynomial = np.poly1d(coefficients)
new_xx = np.linspace(-1.0,1.0)
new_yy = polynomial(new_xx)
ax4.plot(new_xx, new_yy,color = 'r', label = 'y_predict under M = 5')
ax4.legend(loc="upper right")
ax4.set_title('(ii)')
plt.tight_layout() 

# (e)
y_pred_test = poly_lr.predict(poly.fit_transform(df_test[:,:-1]))
print("Part (e) (i)\n")
print("The resulting MSE on the test set: {} \n".format(
    mean_squared_error(df_test[:,-1], y_pred_test)))

## output a csv file
y_pred_unknowns_ = poly_lr.predict(poly.fit_transform(df_unknowns))
np.savetxt("dataset1_unknowns_output_(e).csv", y_pred_unknowns_, delimiter=",", fmt='%f')

"""
Parts (f)-(h) below, use Ridge Regression.

"""
# part (f)
## function that finds the optimal M, lambda
def model_selection_parameter(df_train, df_val):
    results_mse = []
    minimum_mse_on_val = float('inf')
    # potential lambda values
    C = []
    for i in range(0,1300,25):
        C.append(round(0.001*i, 3))
    # potential M values
    degrees = [i for i in range(2,11)]
    ## iterate possible degrees
    for i, degree in enumerate(degrees):
        for j, c in enumerate(C):
            clf = Ridge(alpha= c)
            # generate the expanded feature space
            poly_new = PolynomialFeatures(degree=degree, include_bias=False)     
            # train the model            
            clf.fit(poly_new.fit_transform(df_train[:,:-1]), df_train[:,-1])
            pred = clf.predict(poly_new.fit_transform(df_val[:,:-1]))
            # find the MSE of the current model with degree M, lambda = c
            curr_mse = mean_squared_error(df_val[:,-1], pred)   
            results_mse.append([degree,c,curr_mse])
            # Cross-validation of degree
            if curr_mse < minimum_mse_on_val:
                minimum_mse_on_val = curr_mse
                minimum_degree = degree
                minimum_lambda = c
    return minimum_degree, minimum_lambda, results_mse

clf = Ridge(alpha= 0)
clf.fit(df_train[:,:-1], df_train[:,-1])
# report the MSE on the training set and validation set. 
default_y_pred_train = clf.predict(df_train[:,:-1])
default_y_pred_val = clf.predict(df_val[:,:-1])

print("part f(i) \n")
print("The resulting MSE on the training set using M = 1, Lambda = 0: {} \n".format(
    mean_squared_error(df_train[:,-1], default_y_pred_train)))
print("the resulting MSE on the validation set using M = 1, Lambda = 0: {} \n".format(
    mean_squared_error(df_val[:,-1], default_y_pred_val)))

# get the optimal M, lambda
opt_degree, opt_lambda, mse_results = model_selection_parameter(df_train, df_val)

clf_ridge = Ridge(alpha = opt_lambda)
poly_clf = PolynomialFeatures(degree=opt_degree, include_bias=False)
clf_ridge.fit(poly_clf.fit_transform(df_train[:,:-1]), df_train[:,-1])    
opt_train_weights_M_degree_Lambda_constrains = \
    np.concatenate((np.array([clf_ridge.intercept_]),
                              clf_ridge.coef_))

# report the final weight values
print("The final weight values under optimized degree M = {} and lambda = {} is \n {}".format(
    opt_degree,opt_lambda,opt_train_weights_M_degree_Lambda_constrains.round(decimals=3)))

# report the MSE on the training set and validation set. 
y_pred_train_ = clf_ridge.predict(poly_clf.fit_transform(df_train[:,:-1]))
y_pred_val_ = clf_ridge.predict(poly_clf.fit_transform(df_val[:,:-1]))
print("The resulting MSE on the training set: {} \n".format(
    mean_squared_error(df_train[:,-1], y_pred_train_)))
print("the resulting MSE on the validation set: {} \n".format(
    mean_squared_error(df_val[:,-1], y_pred_val_)))

# part (f) (ii)
mse_results = np.array(mse_results)
res_M_2 = mse_results[np.where(mse_results[:,0] == 2)]
res_M_3 = mse_results[np.where(mse_results[:,0] == 3)]
res_M_4 = mse_results[np.where(mse_results[:,0] == 4)]
res_M_5 = mse_results[np.where(mse_results[:,0] == 5)]
res_M_6 = mse_results[np.where(mse_results[:,0] == 6)]
res_M_7 = mse_results[np.where(mse_results[:,0] == 7)]
res_M_8 = mse_results[np.where(mse_results[:,0] == 8)]
res_M_9 = mse_results[np.where(mse_results[:,0] == 9)]
res_M_10 = mse_results[np.where(mse_results[:,0] == 10)]

## plot each MSE using different lambda uder same M
fig, axs = plt.subplots(3, 3)
fig.suptitle("Validation-set-MSE vs. $\lambda$ plot for each value of M")
axs[1,0].set(ylabel='Validation-set-MSE')
axs[2,1].set(xlabel='$\lambda$') 

axs[0,0].plot(res_M_2[:,1],res_M_2[:,2],label = 'M = 2')
axs[0,0].legend(loc="upper right")

axs[0,1].plot(res_M_3[:,1],res_M_3[:,2],label = 'M = 3')
axs[0,1].legend(loc="upper right")

axs[0,2].plot(res_M_4[:,1],res_M_4[:,2],label = 'M = 4')
axs[0,2].legend(loc="upper right")

axs[1,0].plot(res_M_5[:,1],res_M_5[:,2],label = 'M = 5')
axs[1,0].legend(loc="upper right")

axs[1,1].plot(res_M_6[:,1],res_M_6[:,2],label = 'M = 6')
axs[1,1].legend(loc="upper right")

axs[1,2].plot(res_M_7[:,1],res_M_7[:,2],label = 'M = 7')
axs[1,2].legend(loc="upper right")

axs[2,0].plot(res_M_8[:,1],res_M_8[:,2],label = 'M = 8')
axs[2,0].legend(loc="upper right")

axs[2,1].plot(res_M_9[:,1],res_M_9[:,2],label = 'M = 9')
axs[2,1].legend(loc="upper right")

axs[2,2].plot(res_M_10[:,1],res_M_10[:,2],label = 'M = 10')
axs[2,2].legend(loc="upper right")

plt.tight_layout()
plt.figsize=(20, 20)

"""
Run the system to predict output values on the unknown dataset,

Output a csv file with one column that gives the predicted values
"""
print('Part H (i) \n')
clf_ridge = Ridge(alpha = opt_lambda)
poly_clf = PolynomialFeatures(degree=opt_degree, include_bias=False)
clf_ridge.fit(poly_clf.fit_transform(df_train[:,:-1]), df_train[:,-1]) 
y_pred_test = clf_ridge.predict(poly_clf.fit_transform(df_test[:,:-1]))
y_pred_unknowns = clf_ridge.predict(poly_clf.fit_transform(df_unknowns))
print("the resulting MSE on the test set: {} \n".format(
    mean_squared_error(df_test[:,-1], y_pred_test)))
## output a csv file
np.savetxt("dataset1_unknowns_output_(h).csv", y_pred_unknowns, delimiter=",", fmt='%f')








