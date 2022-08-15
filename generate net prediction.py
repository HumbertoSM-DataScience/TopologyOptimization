# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:51:33 2022

@author: nevej
"""

#Imports
#Importar bibliotecas

from npy_loader import loader
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

k=95.25
#List of .npy files to be loaded
file_list=['95.25augm_comp_e_dif_comp.npy','95.25augm_top.npy']
#Load the files as input and output of the neural network
loader=loader(file_list)
inp, out= loader.load()
#Split the data into training, validation and test data
# Random state=12 is used to guarantee the same split in different trainings of the network
X_train,X_val_and_test,Y_train,Y_val_and_test=train_test_split(inp,out,test_size=4000,random_state=12)
X_val,X_test,Y_val,Y_test=train_test_split(X_val_and_test,Y_val_and_test,test_size=0.5,random_state=12)


print(X_train.shape)
print(X_val.shape)
# print(X_test.shape)
print(Y_train.shape)
print(Y_val.shape)

name_files=str(k) +'_Sosnovik2channels_transposeconv_binarycrossentropyloss_treino_augmented_=_maxpooling_noearlystop_lr0.001_filtros inicial16_droprate0.1'

#Save path to the files with network info and what is obtained after training the network
save_path_general=path='C:/Users/nevej/Documents/IC/Arquivos redes treinadas/'

path_model=os.path.join(save_path_general,name_files,'Saved networks','pbmodel')
model = tf.keras.models.load_model(path_model)
model.summary()
Y_predict_val=model.predict(X_val)
Y_predict_val2=np.copy(Y_predict_val)
for i in range (len(Y_predict_val2)):
    median=np.median(Y_predict_val2[i,:,:,0])
    Y_predict_val2[i,:,:,0][Y_predict_val2[i,:,:,0] > median] = 1
    Y_predict_val2[i,:,:,0][Y_predict_val2[i,:,:,0] <= median] = 0
np.save('C:/Users/nevej/Documents/IC/g2_pred/X_val.npy',X_val)
np.save('C:/Users/nevej/Documents/IC/g2_pred/Y_val.npy',Y_val)
np.save('C:/Users/nevej/Documents/IC/g2_pred/Y_predict_val2.npy',Y_predict_val2)
np.save('C:/Users/nevej/Documents/IC/g2_pred/Y_predict_val.npy',Y_predict_val)
for i in range(100):
    fig, (ax1,ax2,ax3,ax4)=plt.subplots(1,4)
    
    ax1.imshow(X_val[i,:,:,0],cmap='gray_r')
    ax2.imshow(Y_predict_val[i,:,:,0],cmap='gray_r')
    ax3.imshow(Y_predict_val2[i,:,:,0],cmap='gray_r')
    ax4.imshow(Y_val[i,:,:,0],cmap='gray_r')
