# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:23:15 2022

@author: nevej
"""

# Import libraries and modules
import pickle
import os

class save_pickle:
    def __init__(self, history, train_history_dir):
        self.history=history
        self.train_history_dir=train_history_dir
        name_pickle_hist_val_loss='val_loss.pkl'
        name_pickle_hist_loss='loss.pkl'
        name_pickle_hist_bin_acc='bin_acc.pkl'
        name_pickle_hist_val_bin_acc='val_bin_acc.pkl'
        self.name_complete_hist_val_loss=os.path.join(self.train_history_dir,name_pickle_hist_val_loss)
        self.name_complete_hist_loss=os.path.join(self.train_history_dir,name_pickle_hist_loss)
        self.name_complete_hist_bin_acc=os.path.join(self.train_history_dir,name_pickle_hist_bin_acc)
        self.name_complete_hist_val_bin_acc=os.path.join(self.train_history_dir,name_pickle_hist_val_bin_acc)
        
        self.hist_train_bin_acc=self.history.history['binary_accuracy']
        self.hist_val_bin_acc=self.history.history['val_binary_accuracy']
        self.hist_train_loss=self.history.history['loss']
        self.hist_val_loss=self.history.history['val_loss']
        
    def save(self):
        with open(self.name_complete_hist_val_loss, 'wb') as f:
            pickle.dump(self.hist_val_loss, f)
        with open(self.name_complete_hist_loss, 'wb') as f:
            pickle.dump(self.hist_train_loss, f)
        with open(self.name_complete_hist_bin_acc, 'wb') as f:
            pickle.dump(self.hist_train_bin_acc, f)
        with open(self.name_complete_hist_val_bin_acc, 'wb') as f:
            pickle.dump(self.hist_val_bin_acc, f)