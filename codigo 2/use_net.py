# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:41:44 2022

@author: nevej
"""
#Import libraries and modules
import numpy as np
import os
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU

class use_net_on_data:
    def __init__(self, model, X_val, Y_val,path):
        self.model=model
        self.X_val=X_val
        self.Y_val=Y_val
        self.path=path
    def predict_50(self):
        self.Y_predict_val=self.model.predict(self.X_val)
        for i in range (len(self.Y_predict_val)):
            self.median=np.median(self.Y_predict_val[i,:,:,0])
            self.Y_predict_val[i,:,:,0][self.Y_predict_val[i,:,:,0] > self.median] = 1
            self.Y_predict_val[i,:,:,0][self.Y_predict_val[i,:,:,0] <= self.median] = 0
    def calculate_IoU_and_binacc(self):
        self.predict_50()
        IoU=MeanIoU(num_classes=2)
        IoU.update_state(self.Y_val,self.Y_predict_val)
        self.med_Iou=IoU.result().numpy()
        binacc = BinaryAccuracy()
        binacc.update_state(self.Y_val,self.Y_predict_val)
        self.med_bin_acc=binacc.result().numpy()
        return self.med_Iou, self.med_bin_acc
    def save_bin_e_IoU(self):
        name_save='data.txt'
        name_complete = os.path.join(self.path,name_save)
        #print(nome_completo)
        f = open(name_complete, 'w')
        f.write('bin.acc='+str(self.med_bin_acc)+'\n'+'Iou='+str(self.med_Iou))
        f.close()

        