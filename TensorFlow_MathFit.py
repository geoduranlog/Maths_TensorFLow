#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" ==========TensorFlow simple maths fitting============
@author: Alejandro Duran 

SUMMARY:Solution of simple maths and Building Models (TensorFlow() )
"""
#==Clear everything (like a new shell) - useful in Spyder 
from IPython import get_ipython
get_ipython().magic('reset -sf')



# ====Import Libraries=====
import numpy as np									#basic library - maths
import pandas as pd									#data analysis and manipulation
import matplotlib.pyplot as plt						#data visualization
import tensorflow as tf                            #Tensorflow


#OBJECTIVE: Use TensorFlow to fit (a, b, c) in the equation
#    y = (a * x_1 + b * x_2 + c * x_3) ^ 3

# Create a dataset of 1000 samples. The true coefficients used
# are a=2, b=1, c=0.
rs = np.random.RandomState(627372)
n = 1000
X = rs.randn(n, 3)
N=3
Y = ((2 * X[:, 0] + X[:, 1]) ** N + rs.randn(n))[:, np.newaxis] 



#-- I prefer to deal with the data as a dataframe--
Mtx=np.concatenate((X[:, 0].reshape((n,1)),X[:, 1].reshape((n,1)),X[:, 2].reshape((n,1)),Y),axis=1)
df=pd.DataFrame(Mtx) #Create dframe

names = ["x1","x2","x3","Y"]
df.columns=names



#Split into train (80%) and test (20%)  
#If it works well on the test data  (i.e. data the model has not seen during training) there's no need to split into a validation set
train_df=df.sample(frac=0.8, random_state=0)
test_df=df.drop(train_df.index)


#Check statistics and see ranges 
train_df.describe().transpose() #[['mean','std']] 


# Set variables (coefficients to find)
def init():
    A = tf.Variable(1.0) 
    B = tf.Variable(1.0) 
    C = tf.Variable(1.0) 
    
    return A, B, C
A, B, C = init()


#================Model==============
#Instead of using a DNN, I didn't use several layers
#I simply set my loss function as the MAE between y and y_predict  and output coefficients of y_predict
sv_loss=[]
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
for epoch in range(100):
    opt.minimize(lambda: sum(abs(pd.Series.to_numpy(train_df['Y']) - (A*train_df['x1']+ B*train_df['x2']+ C*train_df['x3'])**N)), var_list=[A, B, C])
    sv=sum(abs(pd.Series.to_numpy(train_df['Y']) - (A*train_df['x1']+ B*train_df['x2']+ C*train_df['x3'])**N))
    sv_loss.append(sv)
    print("Epoch: ",epoch)
print()
print("A_pred=",A.numpy())
print("B_pred=",B.numpy())
print("C_pred=",C.numpy())



#Plot loss function with epoch
fig = plt.figure()
plt.plot(list(range(0, epoch+1)), sv_loss,'k.-')
plt.xlabel('Epoch ')
plt.ylabel('Loss')
plt.grid(True)




#--Performance of the Model on Test data--
y_pred=(A*test_df['x1']+ B*test_df['x2']+ C*test_df['x3'])**N



#Compare Test Values with Predictions
fig = plt.figure()
a = plt.axes(aspect='equal')
plt.scatter(test_df['Y'], y_pred)
plt.xlabel('Y True Values ')
plt.ylabel('Y Predictions')
#lims = [min(test_df['Y']), max(test_df['Y'])]
lims = [-100,100]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
    

#Results I got in my laptop
#A_pred= 2.0143945  ~ 2 
#B_pred= 0.9778897  ~ 1
#C_pred= -0.039788753  ~0
