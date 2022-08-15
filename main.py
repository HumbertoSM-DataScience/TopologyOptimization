# -*- coding: utf-8 -*-
"""
Created on Sun May 29 16:56:33 2022

@author: nevej
"""

#Imports
#Importar bibliotecas
import numpy as np
from sklearn.model_selection import train_test_split
#Importar coisas do tensorflow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
#Importar minhas classes
from create_folders_with_net_info import create_folders
from Data_generator import DataGenerator
#from Unet_Sosnovik_2channels import Net
from Plot_and_predict_during_learning import PlotLearning
from Save_hist import Save_hist
from save_pickle import save_pickle
from use_net import use_net_on_data
from save_trained_net import save_trained_net
from Unet_Sosnovik_1channel import Net
#from Plot_and_predict_during_learning import PlotLearning

# Network parameters
#Learning rate
lr=0.001
#Batch size
bs=800
#Number of epochs of training
n_ep=100
#Mass percentage
k=95.25
# Number of convolution filters on the first layers of U-Net
filters_in=16
#Dropout rate
drop_rate=0.1
#Type of network
Net_type='Net_Sosnovik_1channel'
#Name of file to be saved with network info and what is obtained after training the network
# name_files='losscustomizada2'
name_files=str(k) +'round2_sosnovik_complianceclip5timesmedian_1channel_Upsampling2D_batch800_100eps_lr0.001'

#Save path to the files with network info and what is obtained after training the network
save_path_general=path='C:/Users/nevej/Documents/IC/Gen_Arquivos_redes_treinadas/'

# Create folders to save the data of the neural network that will be trained
# Also saves information about the network, such as learning rate and batch size
c=create_folders(save_path_general, lr, bs, n_ep, k, filters_in, drop_rate, name_files, Net_type)
c.create_dirs()
Train_gen = DataGenerator('Training',batch_size=bs)
Val_gen = DataGenerator('Validation',batch_size=bs)
Test_gen = DataGenerator('Test',batch_size=bs)

# Get validation data separated in X_val and Y_val

for k in range(Val_gen.__len__()):
    (inputs,targets)=Val_gen.__getitem__(k)
    if k==0:
        X_val2=inputs
        Y_val2=targets
    else:
        X_val=np.concatenate((X_val2,inputs),axis=0)
        Y_val=np.concatenate((Y_val2,targets),axis=0)
        X_val2=X_val
        Y_val2=Y_val
    


# Build the neural network
net=Net(filters_in, drop_rate)
model=net.build()
model.summary()

#Compile the network
model.compile(optimizer=Adam(learning_rate=lr),  loss='binary_crossentropy', metrics=[BinaryAccuracy()])

# Get paths of directories that were created with create_folder in the beggining of the program
model_dir, train_history_dir, bin_e_IoUs_dir, graphs_dir, info_rede_dir, savesnets_dir=c.get_directories()

#Path to the checkpoints that will be saved during training
checkpoint_path = model_dir+"/Checkpoints/.ckpt"
#model.load_weights(checkpoint_path)
# Create a callback that saves the model's weights

# Model checkpoint callback
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                  monitor='val_binary_accuracy',save_best_only=True,save_weights_only=True,
                                                  verbose=1)
# Early stopping callback
earlystop=EarlyStopping(monitor='val_loss', patience=10,verbose=1,mode='min',restore_best_weights=True)

#Callbacks that will be used are placed in callbacks_list
callbacks_list = [cp_callback,PlotLearning(X_val,Y_val)]

# Fit the model to the data
history=model.fit(Train_gen,
          epochs=n_ep,
          verbose=1,callbacks=callbacks_list,validation_data=(X_val,Y_val))

#loss,acc = model.evaluate(X_val,  Y_val, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100*acc))
#Load weights from checkpoint_path. They are the weights that led to the best validation binary accuracy during training
model.load_weights(checkpoint_path)
# Save the neural network in formats .h5, pbmodel and .onnx
save_net=save_trained_net(savesnets_dir, model)
save_net.save()

# Use trained network on data to predict outputs,
# The 50 % bigger outputs are set to one and the rest to zero
# Calculate the mean IoU and mean binary accuracy on the data and save it in the correcy directory

usenet=use_net_on_data(model, X_val, Y_val,bin_e_IoUs_dir)
mean_Iou,mean_bin_acc=usenet.calculate_IoU_and_binacc()
usenet.save_bin_e_IoU()


# Save plots of the training history
save=Save_hist(history, graphs_dir)
save.save()

# Save training history as lists with pickle
save_pkl=save_pickle(history,train_history_dir)
save_pkl.save()