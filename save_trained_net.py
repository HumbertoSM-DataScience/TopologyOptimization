# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:25:05 2022

@author: nevej
"""

import os
from tensorflow import TensorSpec
from tensorflow import float32
import onnx
import tf2onnx
class save_trained_net:
    def __init__(self, savesnets_dir, model):
        self.savesnets_dir=savesnets_dir
        self.model=model
    def save(self):
        hdf5_path=self.savesnets_dir+"/hdf5/model.h5"
        save_model_path=self.savesnets_dir+"/pbmodel"
        onnx_path=self.savesnets_dir+"/onnx"
        os.mkdir(onnx_path)
        input_signature = [TensorSpec([None,40,40,1], float32)]
        onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature, opset=13)
        onnx.save(onnx_model,onnx_path+"/model.onnx")
        self.model.save(hdf5_path)
        self.model.save(save_model_path)
        