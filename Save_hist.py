# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 12:59:47 2022

@author: nevej
"""
#Import modules and libraries
import matplotlib.pyplot as plt
class Save_hist:
    def __init__(self, history, path):
        self.history=history
        self.path=path
    def save(self):
        plt.figure(figsize=(6,4), dpi=150)
        plt.plot(self.history.history['binary_accuracy'])
        plt.plot(self.history.history['val_binary_accuracy'])
        plt.title('Acurácia binária do modelo')
        plt.ylabel('Acurácia Binária')
        plt.xlabel('Época')
        plt.legend(['Treino', 'Validação'], loc='upper left')
        plt.savefig(self.path +'/plot_binacc.png')
        plt.show()
        plt.clf()
        # summarize history for loss
        plt.figure(figsize=(6,4), dpi=150)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Função $\it{loss}$')
        plt.ylabel('$\it{loss}$')
        plt.xlabel('Época')
        plt.legend(['Treino', 'Validação'], loc='upper left')
        plt.savefig(self.path +'/plot_loss.png')
        plt.show()
        
        