# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 12:34:50 2022

@author: nevej
"""

#Imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, UpSampling2D,Dropout,Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose


class Net:
    # Build a neural network like Sosnovik's
    def __init__(self,filters_init,drop_rate):
        # A diferent number of initial filters and dropout rate can be chosen
        self.filters_init=filters_init
        self.drop_rate=drop_rate
        
    
    
    def build(self):
        #Build the neural network and return it as a tensorflow.keras Model
        
        input_ = Input(shape=(40, 40, 1), name='input_tensor')

        # Level 1
        conv1_1 = Conv2D(self.filters_init, (3, 3), padding='same', activation='relu')(input_) 
        conv1_2 = Conv2D(self.filters_init, (3, 3), padding='same', activation='relu')(conv1_1)
    
        #### Level 2
        pool2_1 = MaxPooling2D((2, 2), padding='same')(conv1_2) 
        conv2_1 = Conv2D(2*self.filters_init, (3, 3), padding='same', activation='relu')(pool2_1)
        drop2_1 = Dropout(self.drop_rate)(conv2_1)
        conv2_2 = Conv2D(2*self.filters_init, (3, 3), padding='same', activation='relu')(drop2_1)
    
        ######### Level 3
        pool3_1 = MaxPooling2D((2, 2), padding='same')(conv2_2) 
        conv3_1 = Conv2D(4*self.filters_init, (3, 3), padding='same', activation='relu')(pool3_1)
        conv3_2 = Conv2D(4*self.filters_init, (3, 3), padding='same', activation='relu')(conv3_1)
    
        conv3_3 = Conv2D(4*self.filters_init, (3, 3), padding='same', activation='relu')(conv3_2)
        conv3_4 = Conv2D(4*self.filters_init, (3, 3), padding='same', activation='relu')(conv3_3)
        up3_1 = UpSampling2D()(conv3_4)
    
        #### Level 2
        concat2 = Concatenate(axis=-1)([conv2_2,up3_1])
        conv2_3 = Conv2D(2*self.filters_init, (3, 3), padding='same', activation='relu')(concat2)
        drop2_2 = Dropout(self.drop_rate)(conv2_3)
        conv2_4 = Conv2D(2*self.filters_init, (3, 3), padding='same', activation='relu')(drop2_2)
        up2_1 = UpSampling2D()(conv2_4)
    
        # Level 1
        concat1 = Concatenate(axis=-1)([conv1_2,up2_1])
        conv1_3 = Conv2D(self.filters_init, (3, 3), padding='same', activation='relu')(concat1)
        conv1_4 = Conv2D(self.filters_init, (3, 3), padding='same', activation='relu')(conv1_3)
        output = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(conv1_4)
    
        model = Model(input_, output)
        return model