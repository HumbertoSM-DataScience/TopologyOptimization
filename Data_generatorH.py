# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:46:27 2022

@author: nevej
"""

# Import libraries

import os
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, data_type, data_path, batch_size=32):
        'Initialization'
        self.data_type = data_type
        self.batch_size = batch_size
        self.data_path = data_path
        if self.data_type=='Training':
            self.path_input= self.data_path+'/Train/Input'
            self.path_output= self.data_path+'/Train/Output'
        elif self.data_type=='Validation':
            self.path_input= self.data_path+'/Val/Input'
            self.path_output= self.data_path+'/Val/Output'
        else:
            self.path_input= self.data_path+'/Test/Input'
            self.path_output= self.data_path+'/Test/Output'
          
        self.files_input=os.listdir(self.path_input)
        self.files_output=os.listdir(self.path_output)
        self.on_epoch_end() 
    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.data_type=='Training':
            return int(np.floor(64000 / self.batch_size))
        elif self.data_type=='Validation':
            return int(np.floor(8000 / self.batch_size))
        else:
            return int(np.floor(8000 / self.batch_size))
    def __getitem__(self,index):
        X=np.empty((self.batch_size,40,40,1))
        y=np.empty((self.batch_size,40,40,1))
        for i, ID in enumerate(self.files_input[index*self.batch_size:(index+1)*self.batch_size]):
            # Store sample
            #print(i)
            path_c_nome_input=os.path.join(self.path_input,ID)
            ID_out = ID.replace('inp.npy','out.npy')
            path_c_nome_output=os.path.join(self.path_output,ID_out)
            X_int = np.load(path_c_nome_input)
            X_2_int = X_int.reshape((40,40,1))
            X[i,] = X_2_int

            # Store class
            Y_int = np.load(path_c_nome_output)
            Y_2_int = Y_int.reshape((40,40,1))
            y[i,] = Y_2_int
            
        return X,y
    def on_epoch_end(self):
        self.files_input,self.files_output=shuffle(self.files_input,self.files_output)