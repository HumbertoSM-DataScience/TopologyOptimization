# -*- coding: utf-8 -*-
"""
Created on Sun May 29 16:59:37 2022

@author: nevej
"""

# Import modules and libraries
import os


#Class to create folders with information about the neural network
class create_folders:
    def __init__(self, save_general_path, lr, bs, n_ep, k, filters_in, drop_rate,name_files, Net_type):
        self.save_general_path=save_general_path
        self.lr=lr
        self.bs=bs
        self.bs=bs
        self.n_ep=n_ep
        self.k=k
        self.filters_in=filters_in
        self.drop_rate=drop_rate
        self.name_files=name_files
        self.Net_type=Net_type
    def create_dirs(self):
        self.model_dir=os.path.join(self.save_general_path,self.name_files)
        os.mkdir(self.model_dir)
        folder_trained_nets='Saved networks'
        self.savesnets_dir=os.path.join(self.model_dir,folder_trained_nets)
        os.mkdir(self.savesnets_dir)
        train_hist_file='History'
        self.train_history_dir=os.path.join(self.model_dir,train_hist_file)
        os.mkdir(self.train_history_dir)
        bin_e_IoUs='Binary accuracies e IoUs'
        self.bin_e_IoUs_dir=os.path.join(self.model_dir,bin_e_IoUs)
        os.mkdir(self.bin_e_IoUs_dir)
        graphs='Graphs'
        self.graphs_dir=os.path.join(self.model_dir,graphs)
        os.mkdir(self.graphs_dir)
        inf_rede='Network info'
        self.info_rede_dir=os.path.join(self.model_dir,inf_rede)
        os.mkdir(self.info_rede_dir)
        complete_name_info=os.path.join(self.info_rede_dir,'info.txt')
        f = open(complete_name_info, 'w')
        f.write(f'Net type={self.Net_type}\nLearning rate={self.lr}\nBatch size={self.bs}\nEpochs={self.n_ep}\nn filtros inicial={self.filters_in}\ndrop_rate={self.drop_rate}')
        f.close()
    def get_directories(self):
        return self.model_dir, self.train_history_dir, self.bin_e_IoUs_dir, self.graphs_dir, self.info_rede_dir, self.savesnets_dir
        