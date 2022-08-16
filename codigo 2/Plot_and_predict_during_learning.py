# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:24:44 2022

@author: nevej
"""

# Import libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU
class PlotLearning(Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def __init__(self,X_val,Y_val):
        self.X_val=X_val
        self.Y_val=Y_val
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics

        self.Y_predict_val=self.model.predict(self.X_val)
        for i in range (len(self.Y_predict_val)):
            mediana=np.median(self.Y_predict_val[i,:,:,0])
            self.Y_predict_val[i,:,:,0][self.Y_predict_val[i,:,:,0] > mediana] = 1
            self.Y_predict_val[i,:,:,0][self.Y_predict_val[i,:,:,0] <= mediana] = 0
    
        IoU=MeanIoU(num_classes=2)
        IoU.update_state(self.Y_val,self.Y_predict_val)
        med_Iou=IoU.result().numpy()
        binacc = BinaryAccuracy()
        binacc.update_state(self.Y_val,self.Y_predict_val)
        med_bin_acc=binacc.result().numpy()
        print(f"Validation binary accuracy={med_bin_acc}")
        print(f"Validation IoU={med_Iou}")
        
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        num=233
        # axs[1,0].imshow(self.Y_predict_val[num,:,:],cmap='gray_r' )
        # axs[1,1].imshow(self.Y_val[num,:,:],cmap='gray_r' )
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()
        
        