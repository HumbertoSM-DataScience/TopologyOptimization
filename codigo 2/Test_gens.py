# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:25:09 2022

@author: nevej
"""

#Imports
#Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Importar coisas do tensorflow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
#Importar minhas classes
from create_folders_with_net_info import create_folders
from Data_generator import DataGenerator
from Unet_Sosnovik_2channels import Net
#from Plot_learning import PlotLearning
from Plot_and_predict_during_learning import PlotLearning

bs=100
Train_gen = DataGenerator('Training',batch_size=bs)
Val_gen = DataGenerator('Validation',batch_size=bs)
Test_gen = DataGenerator('Test',batch_size=bs)
for i in range(Val_gen.__len__()):
    print(i)
# item=Val_gen.__getitem__(0)
# (inputs,targets)=item
# for i in range(len(inputs)):
#     fig,(ax1,ax2)=plt.subplots(1,2)
#     ax1.imshow(inputs[i,:,:,0],cmap='gray_r')
#     ax2.imshow(targets[i,:,:,0],cmap='gray_r')
    