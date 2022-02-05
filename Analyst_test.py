#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" ==========Analyst  Test============
@author: Alejandro Duran 
Solution of simple maths and Building Models (TensorFlow()"""

##SUMMARY: 

#==Clear everything (like a new shell) - useful in Spyder 
from IPython import get_ipython
get_ipython().magic('reset -sf')



# ====Import Libraries=====
import numpy as np									#basic library - maths
import pandas as pd									#data analysis and manipulation
#import seaborn as sns  								 #data visualization - plots
import matplotlib.pyplot as plt						#data visualization
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing



#Use TensorFlow to fit (a, b, c) in the equation
#    y = (a * x_1 + b * x_2 + c * x_3) ^ 3



# Create a dataset of 1000 samples. The true coefficients used
# are a=2, b=1, c=0.
rs = np.random.RandomState(627372)
n = 1000
X = rs.randn(n, 3)
Y = ((2 * X[:, 0] + X[:, 1]) ** 3 + rs.randn(n))[:, np.newaxis] 
#Y = ((2 * X[:, 0] + X[:, 1]) ** 3 )[:, np.newaxis] 



#Check
i=300
print( (2*X[i, 0]+X[i, 1])**3 )
print( Y[i] )


# # Create random constants
a=np.random.randint(0, high=3, size=(n,1))
b=np.random.randint(0, high=3, size=(n,1))
c=np.random.randint(0, high=3, size=(n,1))





#Create constants
def init():
   
    a = tf.Variable.random.uniform(0, maxval=3) 
    b = tf.Variable(1.0) 
    c = tf.Variable(1.0) 
       
    v = tf.Variable(tf.random.truncated_normal([1000, 1]))
    return a, b, c
a, b, c = init()

Y_pred=(a * X[:, 0] + b*X[:, 1]+ c*X[:, 2]) ** 3
Y_pred=Y_pred.reshape(np.size(Y_pred),1)

tst=tf.reshape(Y_pred,(1000,))

A=np.concatenate((a,b,c,X,Y),axis=1)

df=pd.DataFrame(A) #Create dframe

names = ["a","b","c","Y_pred","x1","x2","x3","Y"]
df.columns=names


#Split into train (80%) and test
train_df=df.sample(frac=0.8, random_state=0)
test_df=df.drop(train_df.index)


#Check statistics and see ranges 
train_df.describe().transpose() #[['mean','std']] 



#Separate the target value (the labels) from the features (rest of variables)
train_features = train_df.copy()
test_features = test_df.copy()



#Separate labels from faetures
train_labels=train_features[['a','b','c']]
train_features=train_features[['x1','x2','x3','Y']]

test_labels=test_features[['a','b','c']]
test_features=test_features[['x1','x2','x3','Y']]




#--Define a norm layer (althougth in this case the range of values is not that different)
norm=preprocessing.Normalization()  #Use TensorFLow Keras to create the normalization layer
norm.adapt(np.array(train_features))






#---  MODEL -------:
def build_and_compile_model(train_features):    
#def build_and_compile_model(norm):  
  model = keras.Sequential([
      #norm,
      layers.Dense(2, activation='relu'),
      layers.Dense(6, activation='relu'),
      layers.Dense(3)
  ])
  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
                #optimizer=tf.optimizers.Adam(learning_rate=0.1))
  return model

# #Using all inputs of the train_features
#model = build_and_compile_model(norm)
model = build_and_compile_model(train_features)
#model.summary()



#Train the model
history = model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=1, epochs=1000)

model.summary()


#----Function to create figure
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0.1, 0.8])
  plt.xlabel('Epoch')
  plt.ylabel('Error []')
  plt.legend()
  plt.grid(True)
  
plot_loss(history)



test_predictions = model.predict(test_features) #.flatten()
#namepred = ["a_pred","b_pred","c_pred"]
#test_predictions.columns=namepred

test_pred=test_predictions.mean(axis=0)

test_labels.mean(axis=0)

a = plt.axes(aspect='equal')
#plt.scatter(test_labels['a'], test_predictions[:,1])
plt.scatter(test_labels['a'], test_predictions[:,1])
plt.xlabel('True Values [coef]')
plt.ylabel('Predictions [coef]')
lims = [0, 2]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

