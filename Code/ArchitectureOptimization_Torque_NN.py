#!/usr/bin/env python
# coding: utf-8

# Modules
import constants as const
import numpy as np
from scipy import stats
import pandas as pd
import sciann as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statistics
import time

# Read number of layers and neurons
import sys
#for i in range(1, len(sys.argv)):
    #print('argument:', i, 'value:', sys.argv[i])

N_layers = int(sys.argv[1])
N_neurons = int(sys.argv[2])
#print("N_layers: "+str(N_layers))
#print("N_neurons: "+str(N_neurons))

architecture_string = str(N_layers)+'layers-'+str(N_neurons)+'neurons'
log_file = 'logfile-'+architecture_string+'.txt'
logfile = open(log_file,'w')
logfile.write("N_layers: "+str(N_layers)+'\n')
logfile.write("N_neurons: "+str(N_neurons)+'\n')
logfile.write('\n')

# Read reduced dataset
data_turbines = pd.read_csv('../Dataset/Dataset_reduced.csv')
#print(data_turbines.shape)
#print("Dataset size: "+str(len(data_turbines)))
logfile.write("Dataset size: "+str(len(data_turbines))+'\n')
logfile.write('\n')

# Define inputs and outputs
X_data = data_turbines[['V','theta','W']]
Y_data = data_turbines[['T','P']]
X_data = X_data.to_numpy(dtype='float64')
Y_data = Y_data.to_numpy(dtype='float64')

# Transform inputs and outputs to normalized units, with its mean and deviation
x1_avrg = np.average(X_data[:,0])
x1_std = np.std(X_data[:,0])
x2_avrg = np.average(X_data[:,1])
x2_std = np.std(X_data[:,1])
x3_avrg = np.average(X_data[:,2])
x3_std = np.std(X_data[:,2])
y1_avrg = np.average(Y_data[:,0])
y1_std = np.std(Y_data[:,0])
y2_avrg = np.average(Y_data[:,1])
y2_std = np.std(Y_data[:,1])

X_data[:,0] = ( X_data[:,0] - x1_avrg ) / x1_std
X_data[:,1] = ( X_data[:,1] - x2_avrg ) / x2_std
X_data[:,2] = ( X_data[:,2] - x3_avrg ) / x3_std
Y_data[:,0] = ( Y_data[:,0] - y1_avrg ) / y1_std
Y_data[:,1] = ( Y_data[:,1] - y2_avrg ) / y2_std

# Set up the NN arquitecture to fit the data
from pickletools import optimize
V = sn.Variable('V')
theta = sn.Variable('theta')
W = sn.Variable('W')
# NN arquitecture
architecture = []
#N_layers = 2
#N_neurons = 20
for i in range(N_layers):
    architecture.append(N_neurons)
#Torque = sn.Functional('Torque',variables=[V,theta,W],hidden_layers=[30,30],activation='tanh')
Torque = sn.Functional('Torque',variables=[V,theta,W],hidden_layers=architecture,activation='tanh')

# Set up the optimization algorithm
data1 = sn.Data(Torque)
model = sn.SciModel(inputs=[V,theta,W], targets=data1,loss_func='mae',optimizer='adam')
model.summary()

# Split the data in training and test/validation sets
from tabnanny import verbose
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=38)
train_size = len(X_train[:,0])
test_size = len(X_test[:,0])

# Train the model
start_time = time.time()
#_learning_rate = 0.02
_learning_rate = 0.02
_epochs = 10
#_epochs = 1000
_batch_size = 256
logfile.write("epochs: "+str(_epochs)+'\n')
logfile.write("learning_rate: "+str(_learning_rate)+'\n')
logfile.write("batch_size: "+str(_batch_size)+'\n')
logfile.write('\n')
running = model.train([X_train[:,0],X_train[:,1],X_train[:,2]],Y_train[:,0],batch_size=_batch_size,learning_rate=_learning_rate, epochs=_epochs,verbose=2)
end_time = time.time()
print("END TRAINING \n")

# save model weights 
model.save_weights('weights-'+architecture_string+'.hdf5')

# Save loss
loss_file = 'loss-'+architecture_string+'.dat'
lossfile = open(loss_file,'w')

for i in range(len(running.history['loss'])):
    line = str(i+1)+"  "+str(running.history['loss'][i])+"\n"
    lossfile.write(line)
lossfile.close()

# Evaluate the model
#from tkinter import Y
# mean absolute percentage error (MAPE)
def mape(ytrue, ypred):
    ytrue, ypred = np.array(ytrue), np.array(ypred)
    
    # Remove low values to compute MAPE
    q_low = np.quantile(ytrue,0.01)
    ypred = ypred[ ytrue > q_low ]
    ytrue = ytrue[ ytrue > q_low ]

    mape_i = np.zeros(len(ytrue))
    mape_i = np.abs((ypred-ytrue)/ytrue)*100
    
    return np.mean(mape_i)
    #return np.mean(np.abs((ytrue - ypred) / ytrue)) * 100

# Loss function on test set

y_exact = np.zeros(test_size)
y_pred = np.zeros(test_size)
y_exact[:] = Y_test[:,0]
y_pred[:] = Torque.eval([X_test[:,0],X_test[:,1],X_test[:,2]])

logfile.write("Normalized test MAE: "+str(mean_absolute_error(y_exact,y_pred))+'\n')
logfile.write("Normalized test MSE: "+str(mean_squared_error(y_exact,y_pred))+'\n')
logfile.write("Normalized test MAPE: "+str(mape(y_exact,y_pred))+'\n')
logfile.write('\n')

# Loss function on training set
y_exact = np.zeros(train_size)
y_pred = np.zeros(train_size)
y_exact[:] = Y_train[:,0]
y_pred[:] = Torque.eval([X_train[:,0],X_train[:,1],X_train[:,2]])

logfile.write("Normalized train MAE: "+str(mean_absolute_error(y_exact,y_pred))+'\n')
logfile.write("Normalized train MSE: "+str(mean_squared_error(y_exact,y_pred))+'\n')
logfile.write("Normalized train MAPE: "+str(mape(y_exact,y_pred))+'\n')
logfile.write('\n')

# Evaluation on the original dataset (without normalization)
y_exact = np.zeros(len(Y_data[:,0]))
y_pred = np.zeros(len(Y_data[:,0]))
y_exact[:] = Y_data[:,0] * y1_std + y1_avrg
y_pred[:] = Torque.eval([X_data[:,0],X_data[:,1],X_data[:,2]]) * y1_std + y1_avrg
#error_fit = mean_absolute_error(y_exact,y_pred) / np.mean(y_exact)

logfile.write("Test MAE: "+str(mean_absolute_error(y_exact,y_pred))+'\n')
logfile.write("Test MSE: "+str(mean_squared_error(y_exact,y_pred))+'\n')
logfile.write("Test MAPE: "+str(mape(y_exact,y_pred))+'\n')
logfile.write('\n')

training_time = (end_time - start_time)/60
logfile.write("Training time (min): "+str(training_time)+'\n')